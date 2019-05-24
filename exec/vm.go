package exec

import (
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"math/bits"
	"runtime/debug"
	"strings"
	"time"

	"github.com/go-interpreter/wagon/wasm"
	"github.com/jackc/pgx"
	"github.com/perlin-network/life/compiler"
	"github.com/perlin-network/life/compiler/opcodes"
	"github.com/perlin-network/life/utils"
)

type FunctionImport func(vm *VirtualMachine) int64

const (
	// DefaultCallStackSize is the default call stack size.
	DefaultCallStackSize = 512

	// DefaultPageSize is the linear memory page size.
	DefaultPageSize = 65536

	// JITCodeSizeThreshold is the lower-bound code size threshold for the JIT compiler.
	JITCodeSizeThreshold = 30
)

var (
	// LE is a simple alias to `binary.LittleEndian`.
	LE = binary.LittleEndian

	// Simple counter for operation logging
	opNum int
)

type FunctionImportInfo struct {
	ModuleName string
	FieldName  string
	F          FunctionImport
}

type NCompileConfig struct {
	AliasDef             bool
	DisableMemBoundCheck bool
}

type AOTService interface {
	UnsafeInvokeFunction_0(vm *VirtualMachine, name string) uint64
	UnsafeInvokeFunction_1(vm *VirtualMachine, name string, p0 uint64) uint64
	UnsafeInvokeFunction_2(vm *VirtualMachine, name string, p0, p1 uint64) uint64
}

// VirtualMachine is a WebAssembly execution environment.
type VirtualMachine struct {
	Config           VMConfig
	Module           *compiler.Module
	FunctionCode     []compiler.InterpreterCode
	FunctionImports  []FunctionImportInfo
	CallStack        []Frame
	CurrentFrame     int
	Table            []uint32
	Globals          []int64
	Memory           []byte
	NumValueSlots    int
	Yielded          int64
	InsideExecute    bool
	Delegate         func()
	Exited           bool
	ExitError        interface{}
	ReturnValue      int64
	Gas              uint64
	GasLimitExceeded bool
	GasPolicy        compiler.GasPolicy
	ImportResolver   ImportResolver
	AOTService       AOTService
	StackTrace       string
	pg               *pgx.ConnPool
	PgTx             *pgx.Tx
	PgRunNum         int
}

// VMConfig denotes a set of options passed to a single VirtualMachine instance
type VMConfig struct {
	EnableJIT                bool
	MaxMemoryPages           int
	MaxTableSize             int
	MaxValueSlots            int
	MaxCallStackDepth        int
	DefaultMemoryPages       int
	DefaultTableSize         int
	GasLimit                 uint64
	DisableFloatingPoint     bool
	ReturnOnGasLimitExceeded bool
	PGConfig                 pgx.ConnConfig
	DoOpLogging              bool
}

// Frame represents a call frame.
type Frame struct {
	FunctionID   int
	Code         []byte
	Regs         []int64
	Locals       []int64
	IP           int
	ReturnReg    int
	Continuation int32
}

// ImportResolver is an interface for allowing one to define imports to WebAssembly modules
// ran under a single VirtualMachine instance.
type ImportResolver interface {
	ResolveFunc(module, field string) FunctionImport
	ResolveGlobal(module, field string) int64
}

// NewVirtualMachine instantiates a virtual machine for a given WebAssembly module, with
// specific execution options specified under a VMConfig, and a WebAssembly module import
// resolver.
func NewVirtualMachine(
	code []byte,
	config VMConfig,
	impResolver ImportResolver,
	gasPolicy compiler.GasPolicy,
) (_retVM *VirtualMachine, retErr error) {

	// TODO: Are the DWARF debugging symbols from LLVM useful?

	// If operation logging is enabled, connect to the database
	var err error
	var dbRun int
	var pool *pgx.ConnPool
	if config.DoOpLogging {
		pgPoolConfig := pgx.ConnPoolConfig{config.PGConfig, 45, nil, 5 * time.Second}
		pool, err = pgx.NewConnPool(pgPoolConfig)
		if err != nil {
			panic(err)
		}

		// Grab the next available execution_run number
		dbQuery := `SELECT nextval('execution_runs_seq')`
		err = pool.QueryRow(dbQuery).Scan(&dbRun)
		if err != nil {
			log.Printf("Retrieving next execution run number failed: %v\n", err)
			return nil, err
		}
		log.Printf("opLog execution run: %d\n", dbRun)
	}

	if config.EnableJIT {
		fmt.Println("Warning: JIT support is removed.")
	}

	m, err := compiler.LoadModule(code, pool)
	if err != nil {
		return nil, err
	}

	m.DisableFloatingPoint = config.DisableFloatingPoint

	functionCode, err := m.CompileForInterpreter(gasPolicy)
	if err != nil {
		return nil, err
	}

	defer utils.CatchPanic(&retErr)

	table := make([]uint32, 0)
	globals := make([]int64, 0)
	funcImports := make([]FunctionImportInfo, 0)

	if m.Base.Import != nil && impResolver != nil {
		for _, imp := range m.Base.Import.Entries {
			switch imp.Type.Kind() {
			case wasm.ExternalFunction:
				funcImports = append(funcImports, FunctionImportInfo{
					ModuleName: imp.ModuleName,
					FieldName:  imp.FieldName,
					F:          nil, // deferred
				})
			case wasm.ExternalGlobal:
				globals = append(globals, impResolver.ResolveGlobal(imp.ModuleName, imp.FieldName))
			case wasm.ExternalMemory:
				// TODO: Do we want a real import?
				if m.Base.Memory != nil && len(m.Base.Memory.Entries) > 0 {
					panic("cannot import another memory while we already have one")
				}
				m.Base.Memory = &wasm.SectionMemories{
					Entries: []wasm.Memory{
						wasm.Memory{
							Limits: wasm.ResizableLimits{
								Initial: uint32(config.DefaultMemoryPages),
							},
						},
					},
				}
			case wasm.ExternalTable:
				// TODO: Do we want a real import?
				if m.Base.Table != nil && len(m.Base.Table.Entries) > 0 {
					panic("cannot import another table while we already have one")
				}
				m.Base.Table = &wasm.SectionTables{
					Entries: []wasm.Table{
						wasm.Table{
							Limits: wasm.ResizableLimits{
								Initial: uint32(config.DefaultTableSize),
							},
						},
					},
				}
			default:
				panic(fmt.Errorf("import kind not supported: %d", imp.Type.Kind()))
			}
		}
	}

	// Load global entries.
	for _, entry := range m.Base.GlobalIndexSpace {
		globals = append(globals, execInitExpr(entry.Init, globals))
	}

	// Populate table elements.
	if m.Base.Table != nil && len(m.Base.Table.Entries) > 0 {
		t := &m.Base.Table.Entries[0]

		if config.MaxTableSize != 0 && int(t.Limits.Initial) > config.MaxTableSize {
			panic("max table size exceeded")
		}

		table = make([]uint32, int(t.Limits.Initial))
		for i := 0; i < int(t.Limits.Initial); i++ {
			table[i] = 0xffffffff
		}
		if m.Base.Elements != nil && len(m.Base.Elements.Entries) > 0 {
			for _, e := range m.Base.Elements.Entries {
				offset := int(execInitExpr(e.Offset, globals))
				copy(table[offset:], e.Elems)
			}
		}
	}

	// Load linear memory.
	memory := make([]byte, 0)
	if m.Base.Memory != nil && len(m.Base.Memory.Entries) > 0 {
		initialLimit := int(m.Base.Memory.Entries[0].Limits.Initial)
		if config.MaxMemoryPages != 0 && initialLimit > config.MaxMemoryPages {
			panic("max memory exceeded")
		}

		capacity := initialLimit * DefaultPageSize

		// Initialize empty memory.
		memory = make([]byte, capacity)
		for i := 0; i < capacity; i++ {
			memory[i] = 0
		}

		if m.Base.Data != nil && len(m.Base.Data.Entries) > 0 {
			for _, e := range m.Base.Data.Entries {
				offset := int(execInitExpr(e.Offset, globals))
				copy(memory[int(offset):], e.Data)
			}
		}
	}

	return &VirtualMachine{
		Module:          m,
		Config:          config,
		FunctionCode:    functionCode,
		FunctionImports: funcImports,
		CallStack:       make([]Frame, DefaultCallStackSize),
		CurrentFrame:    -1,
		Table:           table,
		Globals:         globals,
		Memory:          memory,
		Exited:          true,
		GasPolicy:       gasPolicy,
		ImportResolver:  impResolver,
		pg:              pool,
		PgRunNum:        dbRun,
	}, nil
}

func (vm *VirtualMachine) SetAOTService(s AOTService) {
	vm.AOTService = s
}

func bSprintf(builder *strings.Builder, format string, args ...interface{}) {
	builder.WriteString(fmt.Sprintf(format, args...))
}

func escapeName(name string) string {
	ret := ""

	for _, ch := range []byte(name) {
		if (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_' {
			ret += string(ch)
		} else {
			ret += fmt.Sprintf("\\x%02x", ch)
		}
	}

	return ret
}

func filterName(name string) string {
	ret := ""

	for _, ch := range []byte(name) {
		if (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_' {
			ret += string(ch)
		}
	}

	return ret
}

func (vm *VirtualMachine) GenerateNEnv(config NCompileConfig) string {
	builder := &strings.Builder{}

	bSprintf(builder, "#include <stdint.h>\n\n")

	if config.DisableMemBoundCheck {
		builder.WriteString("#define POLYMERASE_NO_MEM_BOUND_CHECK\n")
	}

	builder.WriteString(compiler.NGEN_HEADER)
	if !vm.Config.DisableFloatingPoint {
		builder.WriteString(compiler.NGEN_FP_HEADER)
	}

	bSprintf(builder, "static uint64_t globals[] = {")
	for _, v := range vm.Globals {
		bSprintf(builder, "%dull,", uint64(v))
	}
	bSprintf(builder, "};\n")

	for i, code := range vm.FunctionCode {
		bSprintf(builder, "uint64_t %s%d(struct VirtualMachine *", compiler.NGEN_FUNCTION_PREFIX, i)
		for j := 0; j < code.NumParams; j++ {
			bSprintf(builder, ",uint64_t")
		}
		bSprintf(builder, ");\n")
	}

	// call_indirect dispatcher.
	bSprintf(builder, "struct TableEntry { uint64_t num_params; void *func; };\n")
	bSprintf(builder, "static const uint64_t num_table_entries = %d;\n", len(vm.Table))
	bSprintf(builder, "static struct TableEntry table[] = {\n")
	for _, entry := range vm.Table {
		if entry == math.MaxUint32 {
			bSprintf(builder, "{ .num_params = 0, .func = 0 },\n")
		} else {
			functionID := int(entry)
			code := vm.FunctionCode[functionID]

			bSprintf(builder, "{ .num_params = %d, .func = %s%d },\n", code.NumParams, compiler.NGEN_FUNCTION_PREFIX, functionID)
		}
	}
	bSprintf(builder, "};\n")
	bSprintf(builder, "static void * __attribute__((always_inline)) %sresolve_indirect(struct VirtualMachine *vm, uint64_t entry_id, uint64_t num_params) {\n", compiler.NGEN_ENV_API_PREFIX)
	bSprintf(builder, "if(entry_id >= num_table_entries) { vm->throw_s(vm, \"%s\"); }\n", "table entry out of bounds")
	bSprintf(builder, "if(table[entry_id].func == 0) { vm->throw_s(vm, \"%s\"); }\n", "table entry is null")
	bSprintf(builder, "if(table[entry_id].num_params != num_params) { vm->throw_s(vm, \"%s\"); }\n", "argument count mismatch")
	bSprintf(builder, "return table[entry_id].func;\n")
	bSprintf(builder, "}\n")

	bSprintf(builder, "struct ImportEntry { const char *module_name; const char *field_name; ExternalFunction f; };\n")
	bSprintf(builder, "static const uint64_t num_import_entries = %d;\n", len(vm.FunctionImports))
	bSprintf(builder, "static struct ImportEntry imports[] = {\n")
	for _, imp := range vm.FunctionImports {
		bSprintf(builder, "{ .module_name = \"%s\", .field_name = \"%s\", .f = 0 },\n", escapeName(imp.ModuleName), escapeName(imp.FieldName))
	}
	bSprintf(builder, "};\n")
	bSprintf(builder,
		"static uint64_t __attribute__((always_inline)) %sinvoke_import(struct VirtualMachine *vm, uint64_t import_id, uint64_t num_params, uint64_t *params) {\n",
		compiler.NGEN_ENV_API_PREFIX,
	)

	bSprintf(builder, "if(import_id >= num_import_entries) { vm->throw_s(vm, \"%s\"); }\n", "import entry out of bounds")
	bSprintf(builder, "if(imports[import_id].f == 0) { imports[import_id].f = vm->resolve_import(vm, imports[import_id].module_name, imports[import_id].field_name); }\n")
	bSprintf(builder, "if(imports[import_id].f == 0) { vm->throw_s(vm, \"%s\"); }\n", "cannot resolve import")
	bSprintf(builder, "return imports[import_id].f(vm, import_id, num_params, params);\n")
	bSprintf(builder, "}\n")

	return builder.String()
}

func (vm *VirtualMachine) NBuildAliasDef() string {
	builder := &strings.Builder{}

	builder.WriteString("// Aliases for exported functions\n")

	if vm.Module.Base.Export != nil {
		for name, exp := range vm.Module.Base.Export.Entries {
			if exp.Kind == wasm.ExternalFunction {
				bSprintf(builder, "#define %sexport_%s %s%d\n", compiler.NGEN_FUNCTION_PREFIX, filterName(name), compiler.NGEN_FUNCTION_PREFIX, exp.Index)
			}
		}
	}

	return builder.String()
}

func (vm *VirtualMachine) NCompile(config NCompileConfig) string {
	body, err := vm.Module.CompileWithNGen(vm.GasPolicy, uint64(len(vm.Globals)))
	if err != nil {
		panic(err)
	}

	out := vm.GenerateNEnv(config) + "\n" + body
	if config.AliasDef {
		out += "\n"
		out += vm.NBuildAliasDef()
	}

	return out
}

// Init initializes a frame. Must be called on `call` and `call_indirect`.
func (f *Frame) Init(vm *VirtualMachine, functionID int, code compiler.InterpreterCode) {
	numValueSlots := code.NumRegs + code.NumParams + code.NumLocals
	if vm.Config.MaxValueSlots != 0 && vm.NumValueSlots+numValueSlots > vm.Config.MaxValueSlots {
		panic("max value slot count exceeded")
	}
	vm.NumValueSlots += numValueSlots

	values := make([]int64, numValueSlots)

	f.FunctionID = functionID
	f.Regs = values[:code.NumRegs]
	f.Locals = values[code.NumRegs:]
	f.Code = code.Bytes
	f.IP = 0
	f.Continuation = 0

	//fmt.Printf("Enter function %d (%s)\n", functionID, vm.Module.FunctionNames[functionID])
}

// Destroy destroys a frame. Must be called on return.
func (f *Frame) Destroy(vm *VirtualMachine) {
	numValueSlots := len(f.Regs) + len(f.Locals)
	vm.NumValueSlots -= numValueSlots

	//fmt.Printf("Leave function %d (%s)\n", f.FunctionID, vm.Module.FunctionNames[f.FunctionID])
}

// GetCurrentFrame returns the current frame.
func (vm *VirtualMachine) GetCurrentFrame() *Frame {
	if vm.Config.MaxCallStackDepth != 0 && vm.CurrentFrame >= vm.Config.MaxCallStackDepth {
		panic("max call stack depth exceeded")
	}

	if vm.CurrentFrame >= len(vm.CallStack) {
		panic("call stack overflow")
		//vm.CallStack = append(vm.CallStack, make([]Frame, DefaultCallStackSize / 2)...)
	}
	return &vm.CallStack[vm.CurrentFrame]
}

func (vm *VirtualMachine) getExport(key string, kind wasm.External) (int, bool) {
	if vm.Module.Base.Export == nil {
		return -1, false
	}

	entry, ok := vm.Module.Base.Export.Entries[key]
	if !ok {
		return -1, false
	}

	if entry.Kind != kind {
		return -1, false
	}

	return int(entry.Index), true
}

// GetGlobalExport returns the global export with the given name.
func (vm *VirtualMachine) GetGlobalExport(key string) (int, bool) {
	return vm.getExport(key, wasm.ExternalGlobal)
}

// GetFunctionExport returns the function export with the given name.
func (vm *VirtualMachine) GetFunctionExport(key string) (int, bool) {
	return vm.getExport(key, wasm.ExternalFunction)
}

// PrintStackTrace prints the entire VM stack trace for debugging.
func (vm *VirtualMachine) PrintStackTrace() {
	fmt.Println("--- Begin stack trace ---")
	for i := vm.CurrentFrame; i >= 0; i-- {
		functionID := vm.CallStack[i].FunctionID
		fmt.Printf("<%d> [%d] %s\n", i, functionID, vm.Module.FunctionNames[functionID])
	}
	fmt.Println("--- End stack trace ---")
}

// Ignite initializes the first call frame.
func (vm *VirtualMachine) Ignite(functionID int, params ...int64) {
	if vm.ExitError != nil {
		panic("last execution exited with error; cannot ignite.")
	}

	if vm.CurrentFrame != -1 {
		panic("call stack not empty; cannot ignite.")
	}

	code := vm.FunctionCode[functionID]
	if code.NumParams != len(params) {
		panic("param count mismatch")
	}

	vm.Exited = false

	vm.CurrentFrame++
	frame := vm.GetCurrentFrame()
	frame.Init(
		vm,
		functionID,
		code,
	)
	copy(frame.Locals, params)
}

func (vm *VirtualMachine) AddAndCheckGas(delta uint64) bool {
	newGas := vm.Gas + delta
	if newGas < vm.Gas {
		panic("gas overflow")
	}
	if vm.Config.GasLimit != 0 && newGas > vm.Config.GasLimit {
		if vm.Config.ReturnOnGasLimitExceeded {
			return false
		} else {
			panic("gas limit exceeded")
		}
	}
	vm.Gas = newGas
	return true
}

// Execute starts the virtual machines main instruction processing loop.
// This function may return at any point and is guaranteed to return
// at least once every 10000 instructions. Caller is responsible for
// detecting VM status in a loop.
func (vm *VirtualMachine) Execute() {
	if vm.Exited == true {
		panic("attempting to execute an exited vm")
	}

	if vm.Delegate != nil {
		panic("delegate not cleared")
	}

	if vm.InsideExecute {
		panic("vm execution is not re-entrant")
	}
	vm.InsideExecute = true
	vm.GasLimitExceeded = false

	defer func() {
		vm.InsideExecute = false
		if err := recover(); err != nil {
			vm.Exited = true
			vm.ExitError = err
			vm.StackTrace = string(debug.Stack())
		}
	}()

	frame := vm.GetCurrentFrame()

	for {
		valueID := int(LE.Uint32(frame.Code[frame.IP : frame.IP+4]))
		ins := opcodes.Opcode(frame.Code[frame.IP+4])
		frame.IP += 5

		//fmt.Printf("INS: [%d] %s\n", valueID, ins.String())

		var e error
		switch ins {
		case opcodes.Nop:
			opLog(vm, ins, "Nop", nil, nil)
		case opcodes.Unreachable:
			opLog(vm, ins, "Unreachable", nil, nil)
			panic("wasm: unreachable executed")
		case opcodes.Select:
			if e != nil {
				log.Print(e)
			}
			a := frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]
			b := frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]
			c := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+8:frame.IP+12]))])
			frame.IP += 12
			if c != 0 {
				frame.Regs[valueID] = a
			} else {
				frame.Regs[valueID] = b
			}
			opLog(vm, ins, "Select", []string{"frame_ip", "arg_1", "arg_2", "arg_3", "to_register", "result_value"},
				[]interface{}{frame.IP, a, b, c, valueID, frame.Regs[valueID]})
		case opcodes.I32Const:
			val := LE.Uint32(frame.Code[frame.IP : frame.IP+4])
			frame.IP += 4
			frame.Regs[valueID] = int64(val)
			opLog(vm, ins, "I32 Constant", []string{"frame_ip", "to_register", "result_value"},
				[]interface{}{frame.IP, valueID, val})
		case opcodes.I32Add:
			a := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])
			frame.IP += 8
			val := int64(a + b)
			frame.Regs[valueID] = val
			opLog(vm, ins, "I32 Add", []string{"frame_ip", "base_value", "modifier_value",
				"to_register", "result_value"}, []interface{}{frame.IP, a, b, valueID, val})
		case opcodes.I32Sub:
			a := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])
			frame.IP += 8
			val := int64(a - b)
			frame.Regs[valueID] = val
			opLog(vm, ins, "I32 Subtract", []string{"frame_ip", "base_value", "modifier_value",
				"to_register", "result_value"}, []interface{}{frame.IP, a, b, valueID, val})
		case opcodes.I32Mul:
			a := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])
			frame.IP += 8
			val := int64(a * b)
			frame.Regs[valueID] = val
			opLog(vm, ins, "I32 Multiply", []string{"frame_ip", "base_value", "modifier_value",
				"to_register", "result_value"}, []interface{}{frame.IP, a, b, valueID, val})
		case opcodes.I32DivS:
			a := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])

			if b == 0 {
				opLog(vm, ins, "Panic on I32 Division Signed div by zero", []string{"frame_ip"}, []interface{}{frame.IP})
				panic("integer division by zero")
			}

			if a == math.MinInt32 && b == -1 {
				opLog(vm, ins, "Panic on I32 Division Signed overflow", []string{"frame_ip"}, []interface{}{frame.IP})
				panic("signed integer overflow")
			}

			frame.IP += 8
			val := int64(a / b)
			frame.Regs[valueID] = val
			opLog(vm, ins, "I32 Division Signed", []string{"frame_ip", "base_value", "modifier_value",
				"to_register", "result_value"}, []interface{}{frame.IP, a, b, valueID, val})
		case opcodes.I32DivU:
			a := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])

			if b == 0 {
				opLog(vm, ins, "Panic on I32 Division Unsigned", []string{"frame_ip"}, []interface{}{frame.IP})
				panic("integer division by zero")
			}

			frame.IP += 8
			val := int64(a / b)
			frame.Regs[valueID] = val
			opLog(vm, ins, "I32 Division Unsigned", []string{"frame_ip", "base_value", "modifier_value",
				"to_register", "result_value"}, []interface{}{frame.IP, a, b, valueID, val})
		case opcodes.I32RemS:
			a := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])

			if b == 0 {
				opLog(vm, ins, "Panic on I32 Remainder Signed", []string{"frame_ip"}, []interface{}{frame.IP})
				panic("integer division by zero")
			}

			frame.IP += 8
			val := int64(a % b)
			frame.Regs[valueID] = val
			opLog(vm, ins, "I32 Remainder Signed", []string{"frame_ip", "base_value", "modifier_value",
				"to_register", "result_value"}, []interface{}{frame.IP, a, b, valueID, val})
		case opcodes.I32RemU:
			a := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])

			if b == 0 {
				opLog(vm, ins, "Panic on I32 Remainder Unsigned", []string{"frame_ip"}, []interface{}{frame.IP})
				panic("integer division by zero")
			}

			frame.IP += 8
			val := int64(a % b)
			frame.Regs[valueID] = val
			opLog(vm, ins, "I32 Remainder Unsigned", []string{"frame_ip", "base_value", "modifier_value",
				"to_register", "result_value"}, []interface{}{frame.IP, a, b, valueID, val})
		case opcodes.I32And:
			a := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])

			frame.IP += 8
			val := int64(a & b)
			frame.Regs[valueID] = val
			opLog(vm, ins, "I32 And", []string{"frame_ip", "base_value", "modifier_value",
				"to_register", "result_value"}, []interface{}{frame.IP, a, b, valueID, val})
		case opcodes.I32Or:
			a := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])

			frame.IP += 8
			val := int64(a | b)
			frame.Regs[valueID] = val
			opLog(vm, ins, "I32 Or", []string{"frame_ip", "base_value", "modifier_value",
				"to_register", "result_value"}, []interface{}{frame.IP, a, b, valueID, val})
		case opcodes.I32Xor:
			a := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])

			frame.IP += 8
			val := int64(a ^ b)
			frame.Regs[valueID] = val
			opLog(vm, ins, "I32 Xor", []string{"frame_ip", "base_value", "modifier_value",
				"to_register", "result_value"}, []interface{}{frame.IP, a, b, valueID, val})
		case opcodes.I32Shl:
			a := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])

			frame.IP += 8
			val := int64(a << (b % 32))
			frame.Regs[valueID] = val
			opLog(vm, ins, "I32 Shift left", []string{"frame_ip", "base_value", "modifier_value",
				"to_register", "result_value"}, []interface{}{frame.IP, a, b, valueID, val})
		case opcodes.I32ShrS:
			a := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])

			frame.IP += 8
			val := int64(a >> (b % 32))
			frame.Regs[valueID] = val
			opLog(vm, ins, "I32 Shift right signed", []string{"frame_ip", "base_value", "modifier_value",
				"to_register", "result_value"}, []interface{}{frame.IP, a, b, valueID, val})
		case opcodes.I32ShrU:
			a := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])

			frame.IP += 8
			val := int64(a >> (b % 32))
			frame.Regs[valueID] = val
			opLog(vm, ins, "I32 Shift right unsigned", []string{"frame_ip", "base_value", "modifier_value",
				"to_register", "result_value"}, []interface{}{frame.IP, a, b, valueID, val})
		case opcodes.I32Rotl:
			opLog(vm, ins, "I32Rotl", nil, nil)
			a := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])

			frame.IP += 8
			val := int64(bits.RotateLeft32(a, int(b)))
			frame.Regs[valueID] = val
			opLog(vm, ins, "I32 Rotate left", []string{"frame_ip", "base_value", "modifier_value",
				"to_register", "result_value"}, []interface{}{frame.IP, a, b, valueID, val})
		case opcodes.I32Rotr:
			opLog(vm, ins, "I32Rotr", nil, nil)
			a := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])

			frame.IP += 8
			val := int64(bits.RotateLeft32(a, -int(b)))
			frame.Regs[valueID] = val
			opLog(vm, ins, "I32 Rotate right", []string{"frame_ip", "base_value", "modifier_value",
				"to_register", "result_value"}, []interface{}{frame.IP, a, b, valueID, val})
		case opcodes.I32Clz:
			val := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])

			frame.IP += 4
			result := int64(bits.LeadingZeros32(val))
			frame.Regs[valueID] = result
			opLog(vm, ins, "I32 Count leading zero bits", []string{"frame_ip", "base_value", "to_register",
				"result_value"}, []interface{}{frame.IP, val, valueID, result})
		case opcodes.I32Ctz:
			val := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])

			frame.IP += 4
			result := int64(bits.TrailingZeros32(val))
			frame.Regs[valueID] = result
			opLog(vm, ins, "I32 Count trailing zero bits", []string{"frame_ip", "base_value", "to_register",
				"result_value"}, []interface{}{frame.IP, val, valueID, result})
		case opcodes.I32PopCnt:
			val := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])

			frame.IP += 4
			result := int64(bits.OnesCount32(val))
			frame.Regs[valueID] = result
			opLog(vm, ins, "I32 Count number of one bits", []string{"frame_ip", "base_value", "to_register",
				"result_value"}, []interface{}{frame.IP, val, valueID, result})
		case opcodes.I32EqZ:
			val := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])

			frame.IP += 4
			if val == 0 {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
			opLog(vm, ins, "I32 Equal to Zero", []string{"frame_ip", "base_value", "to_register", "result_value"},
				[]interface{}{frame.IP, val, valueID, frame.Regs[valueID]})

		case opcodes.I32Eq:
			a := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])
			frame.IP += 8
			if a == b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
			opLog(vm, ins, "I32 Equal", []string{"frame_ip", "arg_1", "arg_2", "to_register", "result_value"},
				[]interface{}{frame.IP, a, b, valueID, frame.Regs[valueID]})
		case opcodes.I32Ne:
			a := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])
			frame.IP += 8
			if a != b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
			opLog(vm, ins, "I32 Not Equal", []string{"frame_ip", "arg_1", "arg_2", "to_register", "result_value"},
				[]interface{}{frame.IP, a, b, valueID, frame.Regs[valueID]})
		case opcodes.I32LtS:
			a := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])
			frame.IP += 8
			if a < b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
			opLog(vm, ins, "I32 Less than Signed", []string{"frame_ip", "arg_1", "arg_2", "to_register", "result_value"},
				[]interface{}{frame.IP, a, b, valueID, frame.Regs[valueID]})
		case opcodes.I32LtU:
			opLog(vm, ins, "I32LtU", nil, nil)
			a := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])
			frame.IP += 8
			if a < b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
			opLog(vm, ins, "I32 Less than Unsigned", []string{"frame_ip", "arg_1", "arg_2", "to_register", "result_value"},
				[]interface{}{frame.IP, a, b, valueID, frame.Regs[valueID]})
		case opcodes.I32LeS:
			a := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])
			frame.IP += 8
			if a <= b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
			opLog(vm, ins, "I32 Less than or Equal to Signed", []string{"frame_ip", "arg_1", "arg_2", "to_register", "result_value"},
				[]interface{}{frame.IP, a, b, valueID, frame.Regs[valueID]})
		case opcodes.I32LeU:
			a := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])
			frame.IP += 8
			if a <= b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
			opLog(vm, ins, "I32 Less than or Equal to Unsigned", []string{"frame_ip", "arg_1", "arg_2", "to_register", "result_value"},
				[]interface{}{frame.IP, a, b, valueID, frame.Regs[valueID]})
		case opcodes.I32GtS:
			a := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])
			frame.IP += 8
			if a > b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
			opLog(vm, ins, "I32 Greater than Signed", []string{"frame_ip", "arg_1", "arg_2", "to_register", "result_value"},
				[]interface{}{frame.IP, a, b, valueID, frame.Regs[valueID]})
		case opcodes.I32GtU:
			a := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])
			frame.IP += 8
			if a > b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
			opLog(vm, ins, "I32 Greater than Unsigned", []string{"frame_ip", "arg_1", "arg_2", "to_register", "result_value"},
				[]interface{}{frame.IP, a, b, valueID, frame.Regs[valueID]})
		case opcodes.I32GeS:
			a := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])
			frame.IP += 8
			if a >= b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
			opLog(vm, ins, "I32 Greater than or Equal to Signed", []string{"frame_ip", "arg_1", "arg_2", "to_register", "result_value"},
				[]interface{}{frame.IP, a, b, valueID, frame.Regs[valueID]})
		case opcodes.I32GeU:
			a := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])
			frame.IP += 8
			if a >= b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
			opLog(vm, ins, "I32 Greater than or Equal to Unsigned", []string{"frame_ip", "arg_1", "arg_2", "to_register", "result_value"},
				[]interface{}{frame.IP, a, b, valueID, frame.Regs[valueID]})
		case opcodes.I64Const:
			opLog(vm, ins, "I64Const", nil, nil)
			val := LE.Uint64(frame.Code[frame.IP : frame.IP+8])
			frame.IP += 8
			frame.Regs[valueID] = int64(val)
		case opcodes.I64Add:
			opLog(vm, ins, "I64Add", nil, nil)
			a := frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]
			b := frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]
			frame.IP += 8
			frame.Regs[valueID] = a + b
		case opcodes.I64Sub:
			opLog(vm, ins, "I64Sub", nil, nil)
			a := frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]
			b := frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]
			frame.IP += 8
			frame.Regs[valueID] = a - b
		case opcodes.I64Mul:
			opLog(vm, ins, "I64Mul", nil, nil)
			a := frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]
			b := frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]
			frame.IP += 8
			frame.Regs[valueID] = a * b
		case opcodes.I64DivS:
			opLog(vm, ins, "I64DivS", nil, nil)
			a := frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]
			b := frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]

			if b == 0 {
				panic("integer division by zero")
			}

			if a == math.MinInt64 && b == -1 {
				panic("signed integer overflow")
			}

			frame.IP += 8
			frame.Regs[valueID] = a / b
		case opcodes.I64DivU:
			opLog(vm, ins, "I64DivU", nil, nil)
			a := uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])

			if b == 0 {
				panic("integer division by zero")
			}

			frame.IP += 8
			frame.Regs[valueID] = int64(a / b)
		case opcodes.I64RemS:
			opLog(vm, ins, "I64RemS", nil, nil)
			a := frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]
			b := frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]

			if b == 0 {
				panic("integer division by zero")
			}

			frame.IP += 8
			frame.Regs[valueID] = a % b
		case opcodes.I64RemU:
			opLog(vm, ins, "I64RemU", nil, nil)
			a := uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])

			if b == 0 {
				panic("integer division by zero")
			}

			frame.IP += 8
			frame.Regs[valueID] = int64(a % b)
		case opcodes.I64And:
			opLog(vm, ins, "I64And", nil, nil)
			a := frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]
			b := frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]

			frame.IP += 8
			frame.Regs[valueID] = a & b
		case opcodes.I64Or:
			opLog(vm, ins, "I64Or", nil, nil)
			a := frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]
			b := frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]

			frame.IP += 8
			frame.Regs[valueID] = a | b
		case opcodes.I64Xor:
			opLog(vm, ins, "I64Xor", nil, nil)
			a := frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]
			b := frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]

			frame.IP += 8
			frame.Regs[valueID] = a ^ b
		case opcodes.I64Shl:
			opLog(vm, ins, "I64Shl", nil, nil)
			a := frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]
			b := uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])

			frame.IP += 8
			frame.Regs[valueID] = a << (b % 64)
		case opcodes.I64ShrS:
			opLog(vm, ins, "I64ShrS", nil, nil)
			a := frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]
			b := uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])

			frame.IP += 8
			frame.Regs[valueID] = a >> (b % 64)
		case opcodes.I64ShrU:
			opLog(vm, ins, "I64ShrU", nil, nil)
			a := uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])

			frame.IP += 8
			frame.Regs[valueID] = int64(a >> (b % 64))
		case opcodes.I64Rotl:
			opLog(vm, ins, "I64Rotl", nil, nil)
			a := uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])

			frame.IP += 8
			frame.Regs[valueID] = int64(bits.RotateLeft64(a, int(b)))
		case opcodes.I64Rotr:
			opLog(vm, ins, "I64Rotr", nil, nil)
			a := uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])

			frame.IP += 8
			frame.Regs[valueID] = int64(bits.RotateLeft64(a, -int(b)))
		case opcodes.I64Clz:
			opLog(vm, ins, "I64Clz", nil, nil)
			val := uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])

			frame.IP += 4
			frame.Regs[valueID] = int64(bits.LeadingZeros64(val))
		case opcodes.I64Ctz:
			opLog(vm, ins, "I64Ctz", nil, nil)
			val := uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])

			frame.IP += 4
			frame.Regs[valueID] = int64(bits.TrailingZeros64(val))
		case opcodes.I64PopCnt:
			opLog(vm, ins, "I64PopCnt", nil, nil)
			val := uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])

			frame.IP += 4
			frame.Regs[valueID] = int64(bits.OnesCount64(val))
		case opcodes.I64EqZ:
			opLog(vm, ins, "I64EqZ", nil, nil)
			val := uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])

			frame.IP += 4
			if val == 0 {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
		case opcodes.I64Eq:
			opLog(vm, ins, "I64Eq", nil, nil)
			a := frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]
			b := frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]
			frame.IP += 8
			if a == b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
		case opcodes.I64Ne:
			opLog(vm, ins, "I64Ne", nil, nil)
			a := frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]
			b := frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]
			frame.IP += 8
			if a != b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
		case opcodes.I64LtS:
			opLog(vm, ins, "I64LtS", nil, nil)
			a := frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]
			b := frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]
			frame.IP += 8
			if a < b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
		case opcodes.I64LtU:
			opLog(vm, ins, "I64LtU", nil, nil)
			a := uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])
			frame.IP += 8
			if a < b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
		case opcodes.I64LeS:
			opLog(vm, ins, "I64LeS", nil, nil)
			a := frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]
			b := frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]
			frame.IP += 8
			if a <= b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
		case opcodes.I64LeU:
			opLog(vm, ins, "I64LeU", nil, nil)
			a := uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])
			frame.IP += 8
			if a <= b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
		case opcodes.I64GtS:
			opLog(vm, ins, "I64GtS", nil, nil)
			a := frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]
			b := frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]
			frame.IP += 8
			if a > b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
		case opcodes.I64GtU:
			opLog(vm, ins, "I64GtU", nil, nil)
			a := uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])
			frame.IP += 8
			if a > b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
		case opcodes.I64GeS:
			opLog(vm, ins, "I64GeS", nil, nil)
			a := frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]
			b := frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]
			frame.IP += 8
			if a >= b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
		case opcodes.I64GeU:
			opLog(vm, ins, "I64GeU", nil, nil)
			a := uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			b := uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))])
			frame.IP += 8
			if a >= b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
		case opcodes.F32Add:
			opLog(vm, ins, "F32Add", nil, nil)
			a := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			b := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]))
			frame.IP += 8

			if c := a + b; c != c {
				frame.Regs[valueID] = int64(0x7FC00000)
			} else {
				frame.Regs[valueID] = int64(math.Float32bits(c))
			}

		case opcodes.F32Sub:
			opLog(vm, ins, "F32Sub", nil, nil)
			a := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			b := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]))
			frame.IP += 8

			if c := a - b; c != c {
				frame.Regs[valueID] = int64(0x7FC00000)
			} else {
				frame.Regs[valueID] = int64(math.Float32bits(c))
			}
		case opcodes.F32Mul:
			opLog(vm, ins, "F32Mul", nil, nil)
			a := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			b := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]))
			frame.IP += 8

			if c := a * b; c != c {
				frame.Regs[valueID] = int64(0x7FC00000)
			} else {
				frame.Regs[valueID] = int64(math.Float32bits(c))
			}
		case opcodes.F32Div:
			opLog(vm, ins, "F32Div", nil, nil)
			a := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			b := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]))
			frame.IP += 8

			if c := a / b; c != c {
				frame.Regs[valueID] = int64(0x7FC00000)
			} else {
				frame.Regs[valueID] = int64(math.Float32bits(c))
			}
		case opcodes.F32Sqrt:
			opLog(vm, ins, "F32Sqrt", nil, nil)
			val := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			frame.IP += 4

			if c := float32(math.Sqrt(float64(val))); c != c {
				frame.Regs[valueID] = int64(0x7FC00000)
			} else {
				frame.Regs[valueID] = int64(math.Float32bits(c))
			}
		case opcodes.F32Min:
			opLog(vm, ins, "F32Min", nil, nil)
			a := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			b := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]))
			frame.IP += 8

			if c := float32(math.Min(float64(a), float64(b))); c != c {
				frame.Regs[valueID] = int64(0x7FC00000)
			} else {
				frame.Regs[valueID] = int64(math.Float32bits(c))
			}
		case opcodes.F32Max:
			opLog(vm, ins, "F32Max", nil, nil)
			a := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			b := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]))
			frame.IP += 8

			if c := float32(math.Max(float64(a), float64(b))); c != c {
				frame.Regs[valueID] = int64(0x7FC00000)
			} else {
				frame.Regs[valueID] = int64(math.Float32bits(c))
			}
		case opcodes.F32Ceil:
			opLog(vm, ins, "F32Ceil", nil, nil)
			val := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			frame.IP += 4

			if c := float32(math.Ceil(float64(val))); c != c {
				frame.Regs[valueID] = int64(0x7FC00000)
			} else {
				frame.Regs[valueID] = int64(math.Float32bits(c))
			}
		case opcodes.F32Floor:
			opLog(vm, ins, "F32Floor", nil, nil)
			val := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			frame.IP += 4

			if c := float32(math.Floor(float64(val))); c != c {
				frame.Regs[valueID] = int64(0x7FC00000)
			} else {
				frame.Regs[valueID] = int64(math.Float32bits(c))
			}
		case opcodes.F32Trunc:
			opLog(vm, ins, "F32Trunc", nil, nil)
			val := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			frame.IP += 4

			if c := float32(math.Trunc(float64(val))); c != c {
				frame.Regs[valueID] = int64(0x7FC00000)
			} else {
				frame.Regs[valueID] = int64(math.Float32bits(c))
			}
		case opcodes.F32Nearest:
			opLog(vm, ins, "F32Nearest", nil, nil)
			val := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			frame.IP += 4

			if c := float32(math.RoundToEven(float64(val))); c != c {
				frame.Regs[valueID] = int64(0x7FC00000)
			} else {
				frame.Regs[valueID] = int64(math.Float32bits(c))
			}
		case opcodes.F32Abs:
			opLog(vm, ins, "F32Abs", nil, nil)
			val := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			frame.IP += 4

			if c := float32(math.Abs(float64(val))); c != c {
				frame.Regs[valueID] = int64(0x7FC00000)
			} else {
				frame.Regs[valueID] = int64(math.Float32bits(c))
			}
		case opcodes.F32Neg:
			opLog(vm, ins, "F32Neg", nil, nil)
			val := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			frame.IP += 4

			if c := float32(-val); c != c {
				frame.Regs[valueID] = int64(0x7FC00000)
			} else {
				frame.Regs[valueID] = int64(math.Float32bits(c))
			}
		case opcodes.F32CopySign:
			opLog(vm, ins, "F32CopySign", nil, nil)
			a := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			b := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]))
			frame.IP += 8

			if c := float32(math.Copysign(float64(a), float64(b))); c != c {
				frame.Regs[valueID] = int64(0x7FC00000)
			} else {
				frame.Regs[valueID] = int64(math.Float32bits(c))
			}
		case opcodes.F32Eq:
			opLog(vm, ins, "F32Eq", nil, nil)
			a := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			b := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]))
			frame.IP += 8

			if a == b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
		case opcodes.F32Ne:
			opLog(vm, ins, "F32Ne", nil, nil)
			a := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			b := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]))
			frame.IP += 8
			if a != b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
		case opcodes.F32Lt:
			opLog(vm, ins, "F32Lt", nil, nil)
			a := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			b := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]))
			frame.IP += 8
			if a < b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
		case opcodes.F32Le:
			opLog(vm, ins, "F32Le", nil, nil)
			a := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			b := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]))
			frame.IP += 8
			if a <= b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
		case opcodes.F32Gt:
			opLog(vm, ins, "F32Gt", nil, nil)
			a := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			b := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]))
			frame.IP += 8
			if a > b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
		case opcodes.F32Ge:
			opLog(vm, ins, "F32Ge", nil, nil)
			a := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			b := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]))
			frame.IP += 8
			if a >= b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
		case opcodes.F64Add:
			opLog(vm, ins, "F64Add", nil, nil)
			a := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			b := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]))
			frame.IP += 8

			if c := a + b; c != c {
				frame.Regs[valueID] = int64(0x7FF8000000000001)
			} else {
				frame.Regs[valueID] = int64(math.Float64bits(c))
			}
		case opcodes.F64Sub:
			opLog(vm, ins, "F64Sub", nil, nil)
			a := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			b := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]))
			frame.IP += 8

			if c := a - b; c != c {
				frame.Regs[valueID] = int64(0x7FF8000000000001)
			} else {
				frame.Regs[valueID] = int64(math.Float64bits(c))
			}
		case opcodes.F64Mul:
			opLog(vm, ins, "F64Mul", nil, nil)
			a := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			b := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]))
			frame.IP += 8

			if c := a * b; c != c {
				frame.Regs[valueID] = int64(0x7FF8000000000001)
			} else {
				frame.Regs[valueID] = int64(math.Float64bits(c))
			}
		case opcodes.F64Div:
			opLog(vm, ins, "F64Div", nil, nil)
			a := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			b := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]))
			frame.IP += 8

			if c := a / b; c != c {
				frame.Regs[valueID] = int64(0x7FF8000000000001)
			} else {
				frame.Regs[valueID] = int64(math.Float64bits(c))
			}
		case opcodes.F64Sqrt:
			opLog(vm, ins, "F64Sqrt", nil, nil)
			val := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			frame.IP += 4

			if c := math.Sqrt(val); c != c {
				frame.Regs[valueID] = int64(0x7FF8000000000001)
			} else {
				frame.Regs[valueID] = int64(math.Float64bits(c))
			}
		case opcodes.F64Min:
			opLog(vm, ins, "F64Min", nil, nil)
			a := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			b := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]))
			frame.IP += 8

			if c := math.Min(a, b); c != c {
				frame.Regs[valueID] = int64(0x7FF8000000000001)
			} else {
				frame.Regs[valueID] = int64(math.Float64bits(c))
			}
		case opcodes.F64Max:
			opLog(vm, ins, "F64Max", nil, nil)
			a := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			b := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]))
			frame.IP += 8

			if c := math.Max(a, b); c != c {
				frame.Regs[valueID] = int64(0x7FF8000000000001)
			} else {
				frame.Regs[valueID] = int64(math.Float64bits(c))
			}
		case opcodes.F64Ceil:
			opLog(vm, ins, "F64Ceil", nil, nil)
			val := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			frame.IP += 4

			if c := math.Ceil(val); c != c {
				frame.Regs[valueID] = int64(0x7FF8000000000001)
			} else {
				frame.Regs[valueID] = int64(math.Float64bits(c))
			}
		case opcodes.F64Floor:
			opLog(vm, ins, "F64Floor", nil, nil)
			val := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			frame.IP += 4

			if c := math.Floor(val); c != c {
				frame.Regs[valueID] = int64(0x7FF8000000000001)
			} else {
				frame.Regs[valueID] = int64(math.Float64bits(c))
			}
		case opcodes.F64Trunc:
			opLog(vm, ins, "F64Trunc", nil, nil)
			val := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			frame.IP += 4

			if c := math.Trunc(val); c != c {
				frame.Regs[valueID] = int64(0x7FF8000000000001)
			} else {
				frame.Regs[valueID] = int64(math.Float64bits(c))
			}
		case opcodes.F64Nearest:
			opLog(vm, ins, "F64Nearest", nil, nil)
			val := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			frame.IP += 4

			if c := math.RoundToEven(val); c != c {
				frame.Regs[valueID] = int64(0x7FF8000000000001)
			} else {
				frame.Regs[valueID] = int64(math.Float64bits(c))
			}
		case opcodes.F64Abs:
			opLog(vm, ins, "F64Abs", nil, nil)
			val := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			frame.IP += 4

			if c := math.Abs(val); c != c {
				frame.Regs[valueID] = int64(0x7FF8000000000001)
			} else {
				frame.Regs[valueID] = int64(math.Float64bits(c))
			}
		case opcodes.F64Neg:
			opLog(vm, ins, "F64Neg", nil, nil)
			val := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			frame.IP += 4

			if c := -val; c != c {
				frame.Regs[valueID] = int64(0x7FF8000000000001)
			} else {
				frame.Regs[valueID] = int64(math.Float64bits(c))
			}
		case opcodes.F64CopySign:
			opLog(vm, ins, "F64CopySign", nil, nil)
			a := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			b := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]))
			frame.IP += 8

			if c := math.Copysign(a, b); c != c {
				frame.Regs[valueID] = int64(0x7FF8000000000001)
			} else {
				frame.Regs[valueID] = int64(math.Float64bits(c))
			}
		case opcodes.F64Eq:
			opLog(vm, ins, "F64Eq", nil, nil)
			a := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			b := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]))
			frame.IP += 8
			if a == b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
		case opcodes.F64Ne:
			opLog(vm, ins, "F64Ne", nil, nil)
			a := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			b := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]))
			frame.IP += 8
			if a != b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
		case opcodes.F64Lt:
			opLog(vm, ins, "F64Lt", nil, nil)
			a := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			b := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]))
			frame.IP += 8
			if a < b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
		case opcodes.F64Le:
			opLog(vm, ins, "F64Le", nil, nil)
			a := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			b := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]))
			frame.IP += 8
			if a <= b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
		case opcodes.F64Gt:
			opLog(vm, ins, "F64Gt", nil, nil)
			a := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			b := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]))
			frame.IP += 8
			if a > b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}
		case opcodes.F64Ge:
			opLog(vm, ins, "F64Ge", nil, nil)
			a := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			b := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]))
			frame.IP += 8
			if a >= b {
				frame.Regs[valueID] = 1
			} else {
				frame.Regs[valueID] = 0
			}

		case opcodes.I32WrapI64:
			opLog(vm, ins, "I32WrapI64", nil, nil)
			v := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			frame.IP += 4
			frame.Regs[valueID] = int64(v)

		case opcodes.I32TruncSF32, opcodes.I32TruncUF32:
			opLog(vm, ins, "I32TruncSF32 / I32TruncUF32", nil, nil)
			v := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			frame.IP += 4

			if c := float32(math.Trunc(float64(v))); c != c {
				frame.Regs[valueID] = int64(0x7FC00000)
			} else {
				frame.Regs[valueID] = int64(int32(c))
			}
		case opcodes.I32TruncSF64, opcodes.I32TruncUF64:
			opLog(vm, ins, "I32TruncSF64 / I32TruncUF64", nil, nil)
			v := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			frame.IP += 4

			if c := math.Trunc(v); c != c {
				frame.Regs[valueID] = int64(0x7FC00000)
			} else {
				frame.Regs[valueID] = int64(int32(c))
			}
		case opcodes.I64TruncSF32, opcodes.I64TruncUF32:
			opLog(vm, ins, "I64TruncSF32 / I64TruncUF32", nil, nil)
			v := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			frame.IP += 4

			if c := math.Trunc(float64(v)); c != c {
				frame.Regs[valueID] = int64(0x7FF8000000000001)
			} else {
				frame.Regs[valueID] = int64(c)
			}
		case opcodes.I64TruncSF64, opcodes.I64TruncUF64:
			opLog(vm, ins, "I64TruncSF64 / I64TruncUF64", nil, nil)
			v := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			frame.IP += 4

			if c := math.Trunc(v); c != c {
				frame.Regs[valueID] = int64(0x7FF8000000000001)
			} else {
				frame.Regs[valueID] = int64(c)
			}
		case opcodes.F32DemoteF64:
			opLog(vm, ins, "F32DemoteF64", nil, nil)
			v := math.Float64frombits(uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			frame.IP += 4
			frame.Regs[valueID] = int64(math.Float32bits(float32(v)))

		case opcodes.F64PromoteF32:
			opLog(vm, ins, "F64PromoteF32", nil, nil)
			v := math.Float32frombits(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			frame.IP += 4

			if c := float64(v); c == math.Float64frombits(0x7FF8000000000000) {
				frame.Regs[valueID] = int64(0x7FF8000000000001)
			} else {
				frame.Regs[valueID] = int64(math.Float64bits(c))
			}
		case opcodes.F32ConvertSI32:
			opLog(vm, ins, "F32ConvertSI32", nil, nil)
			v := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			frame.IP += 4
			frame.Regs[valueID] = int64(math.Float32bits(float32(v)))

		case opcodes.F32ConvertUI32:
			opLog(vm, ins, "F32ConvertUI32", nil, nil)
			v := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			frame.IP += 4
			frame.Regs[valueID] = int64(math.Float32bits(float32(v)))

		case opcodes.F32ConvertSI64:
			opLog(vm, ins, "F32ConvertSI64", nil, nil)
			v := int64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			frame.IP += 4
			frame.Regs[valueID] = int64(math.Float32bits(float32(v)))

		case opcodes.F32ConvertUI64:
			opLog(vm, ins, "F32ConvertUI64", nil, nil)
			v := uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			frame.IP += 4
			frame.Regs[valueID] = int64(math.Float32bits(float32(v)))

		case opcodes.F64ConvertSI32:
			opLog(vm, ins, "F64ConvertSI32", nil, nil)
			v := int32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			frame.IP += 4
			frame.Regs[valueID] = int64(int32(math.Float64bits(float64(v))))

		case opcodes.F64ConvertUI32:
			opLog(vm, ins, "F64ConvertUI32", nil, nil)
			v := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			frame.IP += 4
			frame.Regs[valueID] = int64(int32(math.Float64bits(float64(v))))

		case opcodes.F64ConvertSI64:
			opLog(vm, ins, "F64ConvertSI64", nil, nil)
			v := int64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			frame.IP += 4
			frame.Regs[valueID] = int64(math.Float64bits(float64(v)))

		case opcodes.F64ConvertUI64:
			opLog(vm, ins, "F64ConvertUI64", nil, nil)
			v := uint64(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			frame.IP += 4
			frame.Regs[valueID] = int64(math.Float64bits(float64(v)))

		case opcodes.I64ExtendUI32:
			opLog(vm, ins, "I64ExtendUI32", nil, nil)
			v := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))])
			frame.IP += 4
			frame.Regs[valueID] = int64(v)

		case opcodes.I64ExtendSI32:
			opLog(vm, ins, "I64ExtendSI32", nil, nil)
			v := int32(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			frame.IP += 4
			frame.Regs[valueID] = int64(v)

		case opcodes.I32Load, opcodes.I64Load32U:
			offset := LE.Uint32(frame.Code[frame.IP+4 : frame.IP+8])
			base := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+8:frame.IP+12]))])

			frame.IP += 12

			effective := int(uint64(base) + uint64(offset))
			frame.Regs[valueID] = int64(uint32(LE.Uint32(vm.Memory[effective : effective+4])))
			opLog(vm, ins, "I32/64 mem load 32bit unsigned", []string{"frame_ip", "to_register", "result_value", "memory_address"}, []interface{}{frame.IP, valueID, frame.Regs[valueID], effective})
		case opcodes.I64Load32S:
			offset := LE.Uint32(frame.Code[frame.IP+4 : frame.IP+8])
			base := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+8:frame.IP+12]))])

			frame.IP += 12

			effective := int(uint64(base) + uint64(offset))
			frame.Regs[valueID] = int64(int32(LE.Uint32(vm.Memory[effective : effective+4])))
			opLog(vm, ins, "I64 mem load 32bit signed", []string{"frame_ip", "to_register", "result_value", "memory_address"}, []interface{}{frame.IP, valueID, frame.Regs[valueID], effective})
		case opcodes.I64Load:
			offset := LE.Uint32(frame.Code[frame.IP+4 : frame.IP+8])
			base := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+8:frame.IP+12]))])

			frame.IP += 12

			effective := int(uint64(base) + uint64(offset))
			frame.Regs[valueID] = int64(LE.Uint64(vm.Memory[effective : effective+8]))
			opLog(vm, ins, "I64 mem load 64bit", []string{"frame_ip", "to_register", "result_value", "memory_address"}, []interface{}{frame.IP, valueID, frame.Regs[valueID], effective})
		case opcodes.I32Load8S, opcodes.I64Load8S:
			offset := LE.Uint32(frame.Code[frame.IP+4 : frame.IP+8])
			base := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+8:frame.IP+12]))])

			frame.IP += 12

			effective := int(uint64(base) + uint64(offset))
			frame.Regs[valueID] = int64(int8(vm.Memory[effective]))
			opLog(vm, ins, "I32/64 mem load 8bit signed", []string{"frame_ip", "to_register", "result_value", "memory_address"}, []interface{}{frame.IP, valueID, frame.Regs[valueID], effective})
		case opcodes.I32Load8U, opcodes.I64Load8U:
			offset := LE.Uint32(frame.Code[frame.IP+4 : frame.IP+8])
			base := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+8:frame.IP+12]))])

			frame.IP += 12

			effective := int(uint64(base) + uint64(offset))
			frame.Regs[valueID] = int64(uint8(vm.Memory[effective]))
			opLog(vm, ins, "I32/64 mem load 8bit unsigned", []string{"frame_ip", "to_register", "result_value", "memory_address"}, []interface{}{frame.IP, valueID, frame.Regs[valueID], effective})
		case opcodes.I32Load16S, opcodes.I64Load16S:
			offset := LE.Uint32(frame.Code[frame.IP+4 : frame.IP+8])
			base := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+8:frame.IP+12]))])

			frame.IP += 12

			effective := int(uint64(base) + uint64(offset))
			frame.Regs[valueID] = int64(int16(LE.Uint16(vm.Memory[effective : effective+2])))
			opLog(vm, ins, "I32/64 mem load 16bit signed", []string{"frame_ip", "to_register", "result_value", "memory_address"}, []interface{}{frame.IP, valueID, frame.Regs[valueID], effective})
		case opcodes.I32Load16U, opcodes.I64Load16U:
			offset := LE.Uint32(frame.Code[frame.IP+4 : frame.IP+8])
			base := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+8:frame.IP+12]))])

			frame.IP += 12

			effective := int(uint64(base) + uint64(offset))
			frame.Regs[valueID] = int64(uint16(LE.Uint16(vm.Memory[effective : effective+2])))
			opLog(vm, ins, "I32/64 mem load 16bit unsigned", []string{"frame_ip", "to_register", "result_value", "memory_address"}, []interface{}{frame.IP, valueID, frame.Regs[valueID], effective})
		case opcodes.I32Store, opcodes.I64Store32:
			offset := LE.Uint32(frame.Code[frame.IP+4 : frame.IP+8])
			base := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+8:frame.IP+12]))])

			value := frame.Regs[int(LE.Uint32(frame.Code[frame.IP+12:frame.IP+16]))]

			frame.IP += 16

			effective := int(uint64(base) + uint64(offset))
			LE.PutUint32(vm.Memory[effective:effective+4], uint32(value))
			opLog(vm, ins, "I32/64 mem store 32bit", []string{"frame_ip", "memory_address", "result_value"}, []interface{}{frame.IP, effective, uint32(value)})
		case opcodes.I64Store:
			offset := LE.Uint32(frame.Code[frame.IP+4 : frame.IP+8])
			base := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+8:frame.IP+12]))])

			value := frame.Regs[int(LE.Uint32(frame.Code[frame.IP+12:frame.IP+16]))]

			frame.IP += 16

			effective := int(uint64(base) + uint64(offset))
			LE.PutUint64(vm.Memory[effective:effective+8], uint64(value))
			opLog(vm, ins, "I64 mem store 64bit", []string{"frame_ip", "memory_address", "result_value"}, []interface{}{frame.IP, effective, uint64(value)})
		case opcodes.I32Store8, opcodes.I64Store8:
			offset := LE.Uint32(frame.Code[frame.IP+4 : frame.IP+8])
			base := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+8:frame.IP+12]))])

			value := frame.Regs[int(LE.Uint32(frame.Code[frame.IP+12:frame.IP+16]))]

			frame.IP += 16

			effective := int(uint64(base) + uint64(offset))
			vm.Memory[effective] = byte(value)
			opLog(vm, ins, "I32/64 mem store 8bit", []string{"frame_ip", "memory_address", "result_value"},
				[]interface{}{frame.IP, effective, byte(value)})
		case opcodes.I32Store16, opcodes.I64Store16:
			offset := LE.Uint32(frame.Code[frame.IP+4 : frame.IP+8])
			base := uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP+8:frame.IP+12]))])

			value := frame.Regs[int(LE.Uint32(frame.Code[frame.IP+12:frame.IP+16]))]

			frame.IP += 16

			effective := int(uint64(base) + uint64(offset))
			LE.PutUint16(vm.Memory[effective:effective+2], uint16(value))
			opLog(vm, ins, "I32/64 mem store 16bit", []string{"frame_ip", "memory_address", "result_value"},
				[]interface{}{frame.IP, effective, uint16(value)})
		case opcodes.Jmp:
			opLog(vm, ins, "jmp", nil, nil)
			target := int(LE.Uint32(frame.Code[frame.IP : frame.IP+4]))
			vm.Yielded = frame.Regs[int(LE.Uint32(frame.Code[frame.IP+4:frame.IP+8]))]
			frame.IP = target
		case opcodes.JmpEither:
			opLog(vm, ins, "jump either", nil, nil)
			targetA := int(LE.Uint32(frame.Code[frame.IP : frame.IP+4]))
			targetB := int(LE.Uint32(frame.Code[frame.IP+4 : frame.IP+8]))
			cond := int(LE.Uint32(frame.Code[frame.IP+8 : frame.IP+12]))
			yieldedReg := int(LE.Uint32(frame.Code[frame.IP+12 : frame.IP+16]))
			frame.IP += 16

			vm.Yielded = frame.Regs[yieldedReg]
			if frame.Regs[cond] != 0 {
				frame.IP = targetA
			} else {
				frame.IP = targetB
			}
		case opcodes.JmpIf:
			target := int(LE.Uint32(frame.Code[frame.IP : frame.IP+4]))
			cond := int(LE.Uint32(frame.Code[frame.IP+4 : frame.IP+8]))
			yieldedReg := int(LE.Uint32(frame.Code[frame.IP+8 : frame.IP+12]))
			frame.IP += 12
			if frame.Regs[cond] != 0 {
				vm.Yielded = frame.Regs[yieldedReg]
				frame.IP = target
			}
			opLog(vm, ins, "Jump If", []string{"frame_ip", "target", "condition"},
				[]interface{}{frame.IP, target, cond})
		case opcodes.JmpTable:
			opLog(vm, ins, "jump table", nil, nil)
			targetCount := int(LE.Uint32(frame.Code[frame.IP : frame.IP+4]))
			frame.IP += 4

			targetsRaw := frame.Code[frame.IP : frame.IP+4*targetCount]
			frame.IP += 4 * targetCount

			defaultTarget := int(LE.Uint32(frame.Code[frame.IP : frame.IP+4]))
			frame.IP += 4

			cond := int(LE.Uint32(frame.Code[frame.IP : frame.IP+4]))
			frame.IP += 4

			vm.Yielded = frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]
			frame.IP += 4

			val := int(frame.Regs[cond])
			if val >= 0 && val < targetCount {
				frame.IP = int(LE.Uint32(targetsRaw[val*4 : val*4+4]))
			} else {
				frame.IP = defaultTarget
			}
		case opcodes.ReturnValue:
			val := frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]
			frame.Destroy(vm)
			vm.CurrentFrame--
			if vm.CurrentFrame == -1 {
				vm.Exited = true
				vm.ReturnValue = val
				opLog(vm, ins, "VM exit on Return Value", []string{"frame_ip", "result_value"}, []interface{}{frame.IP, val})
				return
			} else {
				frame = vm.GetCurrentFrame()
				frame.Regs[frame.ReturnReg] = val
				opLog(vm, ins, "Return Value", []string{"frame_ip", "to_register", "result_value"}, []interface{}{frame.IP, frame.ReturnReg, val})
				//fmt.Printf("Return value %d\n", val)
			}
		case opcodes.ReturnVoid:
			opLog(vm, ins, "Return Void", nil, nil)
			frame.Destroy(vm)
			vm.CurrentFrame--
			if vm.CurrentFrame == -1 {
				vm.Exited = true
				vm.ReturnValue = 0
				opLog(vm, ins, "VM exit on Return Void", nil, nil)
				return
			} else {
				frame = vm.GetCurrentFrame()
			}
		case opcodes.GetLocal:
			id := int(LE.Uint32(frame.Code[frame.IP : frame.IP+4]))
			val := frame.Locals[id]
			frame.IP += 4
			frame.Regs[valueID] = val
			opLog(vm, ins, "Get Local", []string{"frame_ip", "local_id", "to_register", "result_value"}, []interface{}{frame.IP, id, valueID, val})
			//fmt.Printf("GetLocal %d = %d\n", id, val)
		case opcodes.SetLocal:
			id := int(LE.Uint32(frame.Code[frame.IP : frame.IP+4]))
			frameReg := int(LE.Uint32(frame.Code[frame.IP+4 : frame.IP+8]))
			val := frame.Regs[frameReg]
			frame.IP += 8
			frame.Locals[id] = val
			opLog(vm, ins, "Set Local", []string{"frame_ip", "local_id", "from_register", "result_value"}, []interface{}{frame.IP, id, frameReg, val})
			//fmt.Printf("SetLocal %d = %d\n", id, val)
		case opcodes.GetGlobal:
			fromGlobal := int(LE.Uint32(frame.Code[frame.IP : frame.IP+4]))
			frame.Regs[valueID] = vm.Globals[fromGlobal]
			opLog(vm, ins, "Get Global", []string{"frame_ip", "from_global", "to_register", "result_value"}, []interface{}{frame.IP, fromGlobal, valueID, frame.Regs[valueID]})
			frame.IP += 4
		case opcodes.SetGlobal:
			id := int(LE.Uint32(frame.Code[frame.IP : frame.IP+4]))
			fromRegister := int(LE.Uint32(frame.Code[frame.IP+4 : frame.IP+8]))
			val := frame.Regs[fromRegister]
			frame.IP += 8

			vm.Globals[id] = val
			opLog(vm, ins, "Set Global", []string{"frame_ip", "to_global", "from_register", "result_value"}, []interface{}{frame.IP, id, fromRegister, val})
		case opcodes.Call:
			initialIP := frame.IP
			functionID := int(LE.Uint32(frame.Code[frame.IP : frame.IP+4]))
			frame.IP += 4
			argCount := int(LE.Uint32(frame.Code[frame.IP : frame.IP+4]))
			frame.IP += 4
			argsRaw := frame.Code[frame.IP : frame.IP+4*argCount]
			frame.IP += 4 * argCount

			oldRegs := frame.Regs
			frame.ReturnReg = valueID

			vm.CurrentFrame++
			frame = vm.GetCurrentFrame()
			frame.Init(vm, functionID, vm.FunctionCode[functionID])

			funcName := vm.Module.FunctionNames[functionID]
			//if funcName == "runtime.alloc" {
			//	println("alloc() in progress")
			//}

			opStrings := []string{"frame_ip", "function_name", "to_register", "arg_count"}
			opFields := []interface{}{initialIP, funcName, valueID, argCount}

			for i := 0; i < argCount; i++ {
				val := oldRegs[int(LE.Uint32(argsRaw[i*4:i*4+4]))]
				frame.Locals[i] = val
				opStrings = append(opStrings, fmt.Sprintf("arg_%d", i+1))
				opFields = append(opFields, val)
			}

			opLog(vm, ins, "Call", opStrings, opFields)
			//fmt.Println("Call params =", frame.Locals[:argCount])

		case opcodes.CallIndirect:
			opLog(vm, ins, "call indirect", nil, nil)
			typeID := int(LE.Uint32(frame.Code[frame.IP : frame.IP+4]))
			frame.IP += 4
			argCount := int(LE.Uint32(frame.Code[frame.IP:frame.IP+4])) - 1
			frame.IP += 4
			argsRaw := frame.Code[frame.IP : frame.IP+4*argCount]
			frame.IP += 4 * argCount
			tableItemID := frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]
			frame.IP += 4

			sig := &vm.Module.Base.Types.Entries[typeID]

			functionID := int(vm.Table[tableItemID])
			code := vm.FunctionCode[functionID]

			// TODO: We are only checking CC here; Do we want strict typeck?
			if code.NumParams != len(sig.ParamTypes) || code.NumReturns != len(sig.ReturnTypes) {
				panic("type mismatch")
			}

			oldRegs := frame.Regs
			frame.ReturnReg = valueID

			vm.CurrentFrame++
			frame = vm.GetCurrentFrame()
			frame.Init(vm, functionID, code)
			for i := 0; i < argCount; i++ {
				frame.Locals[i] = oldRegs[int(LE.Uint32(argsRaw[i*4:i*4+4]))]
			}

		case opcodes.InvokeImport:
			importID := int(LE.Uint32(frame.Code[frame.IP : frame.IP+4]))
			frame.IP += 4
			vm.Delegate = func() {
				defer func() {
					if err := recover(); err != nil {
						vm.Exited = true
						vm.ExitError = err
					}
				}()
				imp := vm.FunctionImports[importID]
				if imp.F == nil {
					imp.F = vm.ImportResolver.ResolveFunc(imp.ModuleName, imp.FieldName)
				}
				val := imp.F(vm)
				frame.Regs[valueID] = val
				opLog(vm, ins, "Invoke Import", []string{"frame_ip", "module_name", "function_name", "to_register", "result_value"},
					[]interface{}{frame.IP, imp.ModuleName, imp.FieldName, valueID, val})
				return
			}
			return

		case opcodes.CurrentMemory:
			memPages := int64(len(vm.Memory) / DefaultPageSize)
			frame.Regs[valueID] = memPages
			opLog(vm, ins, "Current Memory", []string{"to_register", "result_value"}, []interface{}{valueID, memPages})

		case opcodes.GrowMemory:
			n := int(uint32(frame.Regs[int(LE.Uint32(frame.Code[frame.IP:frame.IP+4]))]))
			frame.IP += 4

			current := len(vm.Memory) / DefaultPageSize
			if vm.Config.MaxMemoryPages == 0 || (current+n >= current && current+n <= vm.Config.MaxMemoryPages) {
				frame.Regs[valueID] = int64(current)
				vm.Memory = append(vm.Memory, make([]byte, n*DefaultPageSize)...)
			} else {
				frame.Regs[valueID] = -1
			}
			opLog(vm, ins, "Grow Memory", []string{"frame_ip", "base_value", "modifier_value", "to_register",
				"result_value"}, []interface{}{frame.IP, current, n, valueID, frame.Regs[valueID]})

		case opcodes.Phi:
			opLog(vm, ins, "phi", nil, nil)
			frame.Regs[valueID] = vm.Yielded

		case opcodes.AddGas:
			delta := LE.Uint64(frame.Code[frame.IP : frame.IP+8])
			opLog(vm, ins, "add gas", []string{"frame_ip", "before_val_int", "modifier_value", "after_val_int"},
				[]interface{}{frame.IP, vm.Gas, delta, vm.Gas + delta})
			frame.IP += 8
			if !vm.AddAndCheckGas(delta) {
				vm.GasLimitExceeded = true
				return
			}

		case opcodes.FPDisabledError:
			opLog(vm, ins, "floating point disabled", nil, nil)
			panic("wasm: floating point disabled")

		default:
			opLog(vm, opcodes.Unknown, "unknown instruction", nil, nil)
			panic("unknown instruction")
		}
	}
}

// Send the opcode data to the database for post-run analysis.  For now we don't return any error code, just to keep
// the likely bulk code changes somewhat simple
func opLog(vm *VirtualMachine, opCode opcodes.Opcode, opName string, fields []string, data []interface{}) {
	if !vm.Config.DoOpLogging {
		return
	}
	if len(fields) != len(data) {
		log.Print("Mismatching field and data count to opLog()")
		return
	}
	var s, t string
	for i, j := range fields {
		s += ", " + j
		t += fmt.Sprintf(", $%d", 5+i)
	}
	dbQuery := fmt.Sprintf(`
		INSERT INTO execution_run (op_num, run_num, op_code, op_name%s)
		VALUES ($1, $2, $3, $4%s)`, s, t)
	var err error
	var commandTag pgx.CommandTag
	// TODO: Surely there's a better way than this?
	switch len(fields) {
	case 0:
		commandTag, err = vm.PgTx.Exec(dbQuery, opNum, vm.PgRunNum, opCode, opName)
	case 1:
		commandTag, err = vm.PgTx.Exec(dbQuery, opNum, vm.PgRunNum, opCode, opName, data[0])
	case 2:
		commandTag, err = vm.PgTx.Exec(dbQuery, opNum, vm.PgRunNum, opCode, opName, data[0], data[1])
	case 3:
		commandTag, err = vm.PgTx.Exec(dbQuery, opNum, vm.PgRunNum, opCode, opName, data[0], data[1], data[2])
	case 4:
		commandTag, err = vm.PgTx.Exec(dbQuery, opNum, vm.PgRunNum, opCode, opName, data[0], data[1], data[2], data[3])
	case 5:
		commandTag, err = vm.PgTx.Exec(dbQuery, opNum, vm.PgRunNum, opCode, opName, data[0], data[1], data[2], data[3], data[4])
	case 6:
		commandTag, err = vm.PgTx.Exec(dbQuery, opNum, vm.PgRunNum, opCode, opName, data[0], data[1], data[2], data[3], data[4], data[5])
	case 7:
		commandTag, err = vm.PgTx.Exec(dbQuery, opNum, vm.PgRunNum, opCode, opName, data[0], data[1], data[2], data[3], data[4], data[5], data[6])
	default:
		log.Printf("Need to add a case for %d to the opLog() function", len(fields))
		return
	}
	if err != nil {
		log.Print(err)
		return
	}
	if numRows := commandTag.RowsAffected(); numRows != 1 {
		log.Printf("Wrong number of rows (%v) affected when logging an operation: %v\n", numRows, opName)
	}

	// Commit every 10k inserts, so quitting via Ctrl+C keeps the info thus far
	if (opNum % 10000) == 0 {
		err = vm.PgTx.Commit() // Set up an automatic transaction commit
		if err != nil {
			panic(err)
		}
		vm.PgTx, err = vm.pg.Begin()
		if err != nil {
			panic(err)
		}
	}
	opNum++
	return
}
