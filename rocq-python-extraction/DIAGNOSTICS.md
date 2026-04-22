# Rocq -> Python Extraction Diagnostics

Every structured extraction diagnostic has a stable code, a category, and a
one-line remediation. Human-readable errors include the same fields as the JSON
line prefixed by `PYTHON_EXTRACTION_DIAGNOSTIC_JSON:`.

## PYEX001 Persistent Arrays Are Unsupported
Remediation: Avoid Rocq PArray in extracted terms or add an explicit Python remapping.

## PYEX002 Prop Term Used For Computation
Remediation: Move the proof to Prop-only positions or return data in Set/Type before extraction.

## PYEX003 Type Alias Declaration Is Not Emitted
Remediation: Use a concrete inductive/record or add an extraction remapping for the type.

## PYEX004 Axiom Has No Computational Realization
Remediation: Provide an Extract Constant remapping for the axiom before Python extraction.

## PYEX005 Exception Term Cannot Be Emitted As An Expression
Remediation: Keep extracted exceptions at statement position or rewrite the Rocq term to return data.

## PYEX006 Erased Logical Value Reached Runtime
Remediation: Keep the logical argument proof-irrelevant or pass a computational witness in Set/Type.

## PYEX007 Unsupported Custom Match Encoding
Remediation: Give Extract Inductive a Python match function with one thunk per constructor plus the scrutinee.

## PYEX008 Custom Constructor Arity Mismatch
Remediation: Make every custom constructor expression accept the same arguments as the Rocq constructor.

## PYEX009 Unknown Monad Marker
Remediation: Use one of the supported __PYMONAD_* markers or extract the operation normally.

## PYEX010 Monad Marker Arity Mismatch
Remediation: Use the marker with the expected number of computational arguments.

## PYEX011 Record Projection Pattern Is Too Complex
Remediation: Split the nested match into separate matches or bind the record before matching.

## PYEX012 Nested Wildcard Binder Escaped Erasure
Remediation: Name the computational field explicitly or keep the wildcard proof-only.

## PYEX013 Coinductive Packet Is Not Stream-Shaped
Remediation: Expose a one-step destructor or avoid relying on Python iterator synthesis.

## PYEX014 Coinductive Constructor Arity Mismatch
Remediation: Use native coinductive constructors without custom erasure for this extraction.

## PYEX015 Mutual Cofixpoint Shape Is Unsupported
Remediation: Extract a wrapper function around one cofixpoint or split the mutual block.

## PYEX016 Higher-Order Module Signature Is Unsupported
Remediation: Extract the applied module result or simplify the functor signature.

## PYEX017 Module Alias Could Not Be Resolved
Remediation: Extract the canonical module path or make the alias transparent before extraction.

## PYEX018 Module Type With Constraints Is Unsupported
Remediation: Extract the constrained module after elaboration or remove the with-constraint.

## PYEX019 Applicative Functor Cache Key Is Unsupported
Remediation: Pass a first-class module value or extract a non-functorized wrapper.

## PYEX020 Expected An Inductive Or Constructor Reference
Remediation: Report this backend invariant with the extracted declaration and source map.

## PYEX021 Expected An Inductive Reference
Remediation: Report this backend invariant with the extracted declaration and source map.

## PYEX022 Pattern Shorthand Was Not Expanded
Remediation: Report this backend invariant; the printer should expand Pusual before rendering.

## PYEX023 Unexpected Coinductive Constructor Payload
Remediation: Report this backend invariant with the constructor and generated source map.

## PYEX024 Unsupported Primitive Integer Type Alias
Remediation: Keep the integer literal in a computational term or remap the type explicitly.

## PYEX025 Unsupported Primitive Float Type Alias
Remediation: Keep the float literal in a computational term or remap the type explicitly.

## PYEX026 Unsupported Primitive String Type Alias
Remediation: Keep the string literal in a computational term or remap the type explicitly.

## PYEX027 Unknown Type Annotation Shape
Remediation: Use an explicit extracted type remapping or simplify the polymorphic type.

## PYEX028 Function Protocol Annotation Is Too Complex
Remediation: Name the higher-order argument type or specialize the extracted function.

## PYEX029 Generic Constructor Could Not Be Typed
Remediation: Specialize the inductive parameters or add a primitive remapping.

## PYEX030 Logical Inductive Used Computationally
Remediation: Move the inductive to Set/Type or erase the use before extraction.

## PYEX031 Logical Record Used Computationally
Remediation: Separate computational fields into a Set/Type record before extraction.

## PYEX032 Proof-Carrying Pair Leaked Into Python
Remediation: Project the computational component before extraction.

## PYEX033 Unsupported Well-Founded Recursion Shape
Remediation: Expose the structurally recursive helper or simplify the Program Fixpoint obligation shape.

## PYEX034 Local Fixpoint Escaped Statement Context
Remediation: Eta-expand the definition so the local fixpoint appears inside a function body.

## PYEX035 Mutual Recursion Has Erased Selected Function
Remediation: Extract a computational member of the mutual block or remove the proof-only member.

## PYEX036 Unsupported Bytes Literal Encoding
Remediation: Restrict extracted strings to byte strings or provide a Python remapping.

Runtime note: Rocq `String.string` extracts to Python `str` at a UTF-8 text
boundary. A generated pattern match that destructs a Python string raises
`_RocqUtf8BoundaryError` when the split would leave an invalid UTF-8 tail, such
as matching `String _ rest` against a string whose first character is encoded
with multiple bytes.

## PYEX037 Unsupported Float Literal
Remediation: Avoid NaN/infinity payloads that cannot round-trip or remap the constant explicitly.

## PYEX038 Generated Python Identifier Is Invalid
Remediation: Rename the Rocq identifier or add an extraction rename before Python extraction.

## PYEX039 Generated Python Name Collision
Remediation: Rename one Rocq declaration or extract through a module namespace.

## PYEX040 Unclassified Extraction Failure
Remediation: Check the detail field, reduce the Rocq input, and add a catalogue entry for this failure.

## PYEX041 Unsupported Real Number Extraction
Remediation: Use nat, positive, N, Z, or Q for extracted computation; Rocq R has no faithful Python runtime mapping.
