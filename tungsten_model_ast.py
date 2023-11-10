import ast
import re
import typing as t
from pathlib import Path

from typing_extensions import Literal

INPUT_CLASS_NAME = "Input"
FIRST_OPTIONAL_INPUT_FIELD_INDEX = 2
OUTPUT_CLASS_NAME = "Output"
MODEL_CLASS_NAME = "StableDiffusion"
GET_LORAS_METHOD_NAME = "get_loras"
GET_TRIGGER_WORDS_METHOD_NAME = "get_trigger_words"
GET_POSITIVE_PROMPT_CHUNKS_METHOD_NAME = "get_extra_prompt_chunks"
GET_NEGATIVE_PROMPT_CHUNKS_METHOD_NAME = "get_extra_negative_prompt_chunks"
VAE_PATHS_VARIABLE_NAME = "VAE_FILE_PATHS"
SD_PATHS_VARIABLE_NAME = "MODEL_FILES"
RE_INPUT_FIELD_NAME = r"[a-zA-Z][a-zA-Z0-9_]*"


class TungstenModelAST:
    def __init__(
        self,
        orig_tungsten_model_path: Path = Path("tungsten_model.py"),
    ) -> None:
        self.orig_tungsten_model_path = orig_tungsten_model_path
        with orig_tungsten_model_path.open("r") as f:
            source = f.read()
        self.ast = ast.parse(source)

    @property
    def input_def_node(self) -> ast.ClassDef:
        return self._find_class_def_in_body(INPUT_CLASS_NAME)

    @property
    def output_def_node(self) -> ast.ClassDef:
        return self._find_class_def_in_body(OUTPUT_CLASS_NAME)

    @property
    def model_def_node(self) -> ast.ClassDef:
        return self._find_class_def_in_body(MODEL_CLASS_NAME)

    @property
    def get_loras_method_def_node(self) -> ast.FunctionDef:
        return _find_method_in_class_def(self.model_def_node, GET_LORAS_METHOD_NAME)

    @property
    def get_extra_prompt_chunks_method_def_node(self) -> ast.FunctionDef:
        return _find_method_in_class_def(
            self.model_def_node, GET_POSITIVE_PROMPT_CHUNKS_METHOD_NAME
        )

    @property
    def get_extra_negative_prompt_chunks_method_def_node(self) -> ast.FunctionDef:
        return _find_method_in_class_def(
            self.model_def_node, GET_NEGATIVE_PROMPT_CHUNKS_METHOD_NAME
        )

    def add_optional_input_field(
        self,
        name: str,
        typename: Literal["str", "bool", "float", "int"],
        default: t.Any,
        *,
        description: t.Optional[str] = None,
        choices: t.Optional[t.Sequence[t.Union[str, int]]] = None,
        ge: t.Optional[float] = None,
        le: t.Optional[float] = None,
        min_length: t.Optional[int] = None,
        max_length: t.Optional[int] = None,
    ):
        assert typename in [
            "str",
            "bool",
            "float",
            "int",
        ], "Supported types: str, bool, float, int"

        target = ast.Name(id=name, ctx=ast.Store())
        annotation = ast.Name(id=typename, ctx=ast.Load())
        option_keywords = []
        if description is not None:
            option_keywords.append(
                ast.keyword(arg="default", value=ast.Constant(value=default))
            )
        if choices is not None:
            option_keywords.append(
                ast.keyword(
                    arg="choices",
                    value=ast.List(
                        elts=list(ast.Constant(value=c) for c in choices),
                        ctx=ast.Load(),
                    ),
                )
            )
        if ge is not None:
            option_keywords.append(ast.keyword(arg="ge", value=ast.Constant(value=ge)))
        if le is not None:
            option_keywords.append(ast.keyword(arg="le", value=ast.Constant(value=le)))
        if min_length is not None:
            option_keywords.append(
                ast.keyword(arg="min_length", value=ast.Constant(value=min_length))
            )
        if max_length is not None:
            option_keywords.append(
                ast.keyword(arg="max_length", value=ast.Constant(value=max_length))
            )
        value = ast.Call(
            func=ast.Name(id="Option", ctx=ast.Load()),
            args=[],
            keywords=option_keywords,
        )
        assign = ast.AnnAssign(
            target=target, annotation=annotation, value=value, simple=1
        )
        self.input_def_node.body.insert(FIRST_OPTIONAL_INPUT_FIELD_INDEX, assign)

    def add_lora(
        self,
        name: str,
        magnitude: t.Optional[float] = None,
        expr: t.Optional[str] = None,
    ):
        self._add_to_list_ret_node_of_model_method_with_input_arg(
            method_name=GET_LORAS_METHOD_NAME,
            keyword=name,
            magnitude=1.0 if magnitude is None and expr is None else magnitude,
            expr=expr,
        )

    def add_triger_word(
        self,
        value: t.Optional[str] = None,
        magnitude: t.Optional[float] = None,
        expr: t.Optional[str] = None,
    ):
        self._add_to_list_ret_node_of_model_method_with_input_arg(
            method_name=GET_TRIGGER_WORDS_METHOD_NAME,
            keyword=value,
            magnitude=magnitude,
            expr=expr,
        )

    def add_extra_prompt_chunk(
        self,
        value: t.Optional[str] = None,
        magnitude: t.Optional[float] = None,
        expr: t.Optional[str] = None,
    ):
        self._add_to_list_ret_node_of_model_method_with_input_arg(
            method_name=GET_POSITIVE_PROMPT_CHUNKS_METHOD_NAME,
            keyword=value,
            magnitude=magnitude,
            expr=expr,
        )

    def add_extra_negative_prompt_chunk(
        self,
        value: t.Optional[str] = None,
        magnitude: t.Optional[float] = None,
        expr: t.Optional[str] = None,
    ):
        self._add_to_list_ret_node_of_model_method_with_input_arg(
            method_name=GET_NEGATIVE_PROMPT_CHUNKS_METHOD_NAME,
            keyword=value,
            magnitude=magnitude,
            expr=expr,
        )

    def unparse(self) -> str:
        return ast.unparse(self.ast)

    def _find_class_def_in_body(self, name: str) -> ast.ClassDef:
        cdef = None
        for node in self.ast.body:
            if isinstance(node, ast.ClassDef) and node.name == name:
                cdef = node
                break

        assert (
            cdef is not None
        ), f"No class definition in {self.orig_tungsten_model_path}: {name}"

        return cdef

    def _add_to_list_ret_node_of_model_method_with_input_arg(
        self,
        method_name: str,
        keyword: t.Optional[str] = None,
        magnitude: t.Optional[float] = None,
        expr: t.Optional[str] = None,
    ):
        if (keyword is None or not keyword.strip()) and (
            expr is None or not expr.strip()
        ):
            return

        method_node = _find_method_in_class_def(self.model_def_node, method_name)
        orig_ret_nodes = _find_ret_nodes_in_function_def(method_node)
        assert all(
            isinstance(node.value, ast.List) for node in orig_ret_nodes
        ), f"{method_name} should return a list"

        for node in orig_ret_nodes:
            ret_val_node: ast.List = node.value  # type: ignore
            ret_val_node.ctx = ast.Load()

            if magnitude is None and expr is None:
                ret_val_node.elts.append(ast.Constant(value=keyword))
            elif magnitude is not None:
                ret_val_node.elts.append(
                    ast.Tuple(
                        elts=[
                            ast.Constant(value=keyword),
                            ast.Constant(value=magnitude),
                        ],
                        ctx=ast.Load(),
                    )
                )
            else:
                assert expr is not None
                matched_input_field_name = re.search(RE_INPUT_FIELD_NAME, expr)
                input_field_name: t.Optional[str] = None
                if matched_input_field_name:
                    input_field_name = matched_input_field_name[0]
                    expr = re.sub(
                        RE_INPUT_FIELD_NAME, "input." + input_field_name, expr
                    )
                if keyword:
                    ret_val_node.elts.append(
                        ast.Tuple(
                            elts=[
                                ast.Constant(value=keyword),
                                ast.parse(expr),
                            ],
                            ctx=ast.Load(),
                        )
                    )
                elif input_field_name:
                    ret_val_node.elts.append(ast.parse(expr))


def _find_method_in_class_def(class_def: ast.ClassDef, name: str) -> ast.FunctionDef:
    fdef = None
    for node in class_def.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            fdef = node
            break

    assert fdef is not None, f"No method in class {class_def.name}: {name}"
    return fdef


def _find_ret_nodes_in_function_def(
    function_def: ast.FunctionDef,
) -> t.List[ast.Return]:
    ret_nodes = []
    for node in function_def.body:
        if isinstance(node, ast.Return):
            ret_nodes.append(node)

    return ret_nodes


if __name__ == "__main__":
    tast = TungstenModelAST()
    tast.add_optional_input_field(
        name="a",
        typename="str",
        default="default",
        description="AAAAA",
        choices=["a"],
        min_length=1,
        max_length=10,
    )
    print(ast.unparse(tast.ast))
    print(ast.dump(tast.model_def_node.body[2]))
