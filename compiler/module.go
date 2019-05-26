package compiler

import (
	"bytes"
	"debug/dwarf"
	"encoding/binary"
	"fmt"
	"io"
	"strings"

	dwr "github.com/go-delve/delve/pkg/dwarf/reader"
	"github.com/go-interpreter/wagon/disasm"
	"github.com/go-interpreter/wagon/wasm"
	"github.com/go-interpreter/wagon/wasm/leb128"
	"github.com/jackc/pgx"
	"github.com/perlin-network/life/compiler/opcodes"
	"github.com/perlin-network/life/utils"
)

type Module struct {
	Base                 *wasm.Module
	FunctionNames        map[int]string
	DisableFloatingPoint bool
}

type InterpreterCode struct {
	NumRegs    int
	NumParams  int
	NumLocals  int
	NumReturns int
	Bytes      []byte
	JITInfo    interface{}
	JITDone    bool
}

func LoadModule(raw []byte, pool *pgx.ConnPool) (*Module, error) {
	reader := bytes.NewReader(raw)

	m, err := wasm.ReadModule(reader, nil)
	if err != nil {
		return nil, err
	}

	/*err = validate.VerifyModule(m)
	if err != nil {
		return nil, err
	}*/

	functionNames := make(map[int]string)

	// Check for the custom sections generated by LLVM
	data := make(map[string][]byte)
	for _, sec := range m.Customs {
		switch sec.Name {
		case "name":
			extractFunctionNames(sec, functionNames)

		case ".debug_info":
			data["info"] = extractDwarf(sec.Name, sec.RawSection.Bytes)

		case ".debug_macinfo":
			data["macinfo"] = extractDwarf(sec.Name, sec.RawSection.Bytes)

		case ".debug_pubtypes":
			data["pubtypes"] = extractDwarf(sec.Name, sec.RawSection.Bytes)

		case ".debug_ranges":
			data["ranges"] = extractDwarf(sec.Name, sec.RawSection.Bytes)

		case ".debug_abbrev":
			data["abbrev"] = extractDwarf(sec.Name, sec.RawSection.Bytes)

		case ".debug_line":
			data["line"] = extractDwarf(sec.Name, sec.RawSection.Bytes)

		case ".debug_str":
			data["str"] = extractDwarf(sec.Name, sec.RawSection.Bytes)

		case ".debug_pubnames":
			data["pubnames"] = extractDwarf(sec.Name, sec.RawSection.Bytes)
		}
	}

	// Construct a DWARF object from the section data
	d, err := dwarf.New(data["abbrev"], nil, nil, data["info"], data["line"], data["pubnames"], data["ranges"], data["str"])
	if err != nil {
		return nil, err
	}
	foo := dwr.New(d)
	fmt.Printf("%v", foo)

	return &Module{
		Base:          m,
		FunctionNames: functionNames,
	}, nil
}

func (m *Module) CompileWithNGen(gp GasPolicy, numGlobals uint64) (out string, retErr error) {
	defer utils.CatchPanic(&retErr)

	importStubBuilder := &strings.Builder{}
	importTypeIDs := make([]int, 0)
	numFuncImports := 0

	if m.Base.Import != nil {
		for i := 0; i < len(m.Base.Import.Entries); i++ {
			e := &m.Base.Import.Entries[i]
			if e.Type.Kind() != wasm.ExternalFunction {
				continue
			}
			tyID := e.Type.(wasm.FuncImport).Type
			ty := &m.Base.Types.Entries[int(tyID)]

			bSprintf(importStubBuilder, "uint64_t %s%d(struct VirtualMachine *vm", NGEN_FUNCTION_PREFIX, i)
			for j := 0; j < len(ty.ParamTypes); j++ {
				bSprintf(importStubBuilder, ",uint64_t %s%d", NGEN_LOCAL_PREFIX, j)
			}
			importStubBuilder.WriteString(") {\n")
			importStubBuilder.WriteString("uint64_t params[] = {")
			for j := 0; j < len(ty.ParamTypes); j++ {
				bSprintf(importStubBuilder, "%s%d", NGEN_LOCAL_PREFIX, j)
				if j != len(ty.ParamTypes)-1 {
					importStubBuilder.WriteByte(',')
				}
			}
			importStubBuilder.WriteString("};\n")
			bSprintf(importStubBuilder, "return %sinvoke_import(vm, %d, %d, params);\n", NGEN_ENV_API_PREFIX, numFuncImports, len(ty.ParamTypes))
			importStubBuilder.WriteString("}\n")
			importTypeIDs = append(importTypeIDs, int(tyID))
			numFuncImports++
		}
	}

	out += importStubBuilder.String()

	for i, f := range m.Base.FunctionIndexSpace {
		//fmt.Printf("Compiling function %d (%+v) with %d locals\n", i, f.Sig, len(f.Body.Locals))
		d, err := disasm.Disassemble(f, m.Base)
		if err != nil {
			panic(err)
		}
		compiler := NewSSAFunctionCompiler(m.Base, d)
		compiler.CallIndexOffset = numFuncImports
		compiler.Compile(importTypeIDs)
		if m.DisableFloatingPoint {
			compiler.FilterFloatingPoint()
		}
		//if gp != nil {
		//	compiler.InsertGasCounters(gp)
		//}
		//fmt.Println(compiler.Code)
		//fmt.Printf("%+v\n", compiler.NewCFGraph())
		//numRegs := compiler.RegAlloc()
		//fmt.Println(compiler.Code)
		numLocals := 0
		for _, v := range f.Body.Locals {
			numLocals += int(v.Count)
		}
		out += compiler.NGen(uint64(numFuncImports+i), uint64(len(f.Sig.ParamTypes)), uint64(numLocals), numGlobals)
	}

	return
}

func (m *Module) CompileForInterpreter(gp GasPolicy) (_retCode []InterpreterCode, retErr error) {
	defer utils.CatchPanic(&retErr)

	ret := make([]InterpreterCode, 0)
	importTypeIDs := make([]int, 0)

	if m.Base.Import != nil {
		for i := 0; i < len(m.Base.Import.Entries); i++ {
			e := &m.Base.Import.Entries[i]
			if e.Type.Kind() != wasm.ExternalFunction {
				continue
			}
			tyID := e.Type.(wasm.FuncImport).Type
			ty := &m.Base.Types.Entries[int(tyID)]

			buf := &bytes.Buffer{}

			binary.Write(buf, binary.LittleEndian, uint32(1)) // value ID
			binary.Write(buf, binary.LittleEndian, opcodes.InvokeImport)
			binary.Write(buf, binary.LittleEndian, uint32(i))

			binary.Write(buf, binary.LittleEndian, uint32(0))
			if len(ty.ReturnTypes) != 0 {
				binary.Write(buf, binary.LittleEndian, opcodes.ReturnValue)
				binary.Write(buf, binary.LittleEndian, uint32(1))
			} else {
				binary.Write(buf, binary.LittleEndian, opcodes.ReturnVoid)
			}

			code := buf.Bytes()

			ret = append(ret, InterpreterCode{
				NumRegs:    2,
				NumParams:  len(ty.ParamTypes),
				NumLocals:  0,
				NumReturns: len(ty.ReturnTypes),
				Bytes:      code,
			})
			importTypeIDs = append(importTypeIDs, int(tyID))
		}
	}

	numFuncImports := len(ret)
	ret = append(ret, make([]InterpreterCode, len(m.Base.FunctionIndexSpace))...)

	for i, f := range m.Base.FunctionIndexSpace {
		//fmt.Printf("Compiling function %d (%+v) with %d locals\n", i, f.Sig, len(f.Body.Locals))
		d, err := disasm.Disassemble(f, m.Base)
		if err != nil {
			panic(err)
		}
		compiler := NewSSAFunctionCompiler(m.Base, d)
		compiler.CallIndexOffset = numFuncImports
		compiler.Compile(importTypeIDs)
		if m.DisableFloatingPoint {
			compiler.FilterFloatingPoint()
		}
		//if gp != nil {
		//	compiler.InsertGasCounters(gp)
		//}
		//fmt.Println(compiler.Code)
		//fmt.Printf("%+v\n", compiler.NewCFGraph())
		numRegs := compiler.RegAlloc()
		//fmt.Println(compiler.Code)
		numLocals := 0
		for _, v := range f.Body.Locals {
			numLocals += int(v.Count)
		}
		ret[numFuncImports+i] = InterpreterCode{
			NumRegs:    numRegs,
			NumParams:  len(f.Sig.ParamTypes),
			NumLocals:  numLocals,
			NumReturns: len(f.Sig.ReturnTypes),
			Bytes:      compiler.Serialize(),
		}
	}

	return ret, nil
}

// Parses the "name" custom section, extracting the function names present in the file
func extractFunctionNames(sec *wasm.SectionCustom, functionNames map[int]string) {
	r := bytes.NewReader(sec.RawSection.Bytes)
	for {
		m, err := leb128.ReadVarUint32(r)
		if err != nil || (m != 1 && m != 4) {
			break
		}

		// LLVM generated "name" section starts with 4, being the length of the word "name"
		if m == 4 {
			b := make([]byte, 5) // Length of "name" plus the byte before hand giving its length
			if _, err = io.ReadFull(r, b); err != nil {
				panic(err)
			}
		}

		// Length of the remaining data in this section
		payloadLen, err := leb128.ReadVarUint32(r)
		if err != nil {
			panic(err)
		}
		data := make([]byte, int(payloadLen))
		n, err := r.Read(data)
		if err != nil {
			panic(err)
		}
		if n != len(data) {
			panic("len mismatch")
		}
		r := bytes.NewReader(data)
		for {
			// The first value contains the number of functions
			count, err := leb128.ReadVarUint32(r)
			if err != nil {
				break
			}
			for i := 0; i < int(count); i++ {
				// Each function name entry contains:
				//   * Its function number (eg 0, 1, 2, etc)
				//   * The length of the name in bytes. eg 10 for "wasm_stuff"
				//   * The name of the function
				index, err := leb128.ReadVarUint32(r)
				if err != nil {
					panic(err)
				}
				nameLen, err := leb128.ReadVarUint32(r)
				if err != nil {
					panic(err)
				}
				name := make([]byte, int(nameLen))
				n, err := r.Read(name)
				if err != nil {
					panic(err)
				}
				if n != len(name) {
					panic("len mismatch")
				}
				functionNames[int(index)] = string(name)
			}
		}
	}
}

// Returns the DWARF data contained in a given custom section
func extractDwarf(name string, data []byte) []byte {
	// Skip past the section name string at the start
	r := bytes.NewReader(data)
	var err error
	b := make([]byte, len(name)+1)
	if _, err = io.ReadFull(r, b); err != nil {
		panic(err)
	}

	// The remaining data should be the DWARF info
	var z bytes.Buffer
	if _, err = io.Copy(&z, r); err != nil {
		panic(err)
	}
	return z.Bytes()
}
