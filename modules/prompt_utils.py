import re

RE_PLAIN_KEYWORD = (
    r"((?:[,\s]+|[\s]+|^)[\(]*{keyword}(?::[0-9]*(?:.[0-9])*)*[\)]*)([,\s]+|[\s]+|$)"
)
RE_LORA_KEYWORD = r"((?:[,\s]+|[\s]+|^)<[\s]*lora[\s]*:[\s]*{keyword}[\s]*(?::[\s]*[\-]*[0-9]*(?:.[0-9]*))*[\s]*>)([,\s]+|\s+|$)"  # noqa: E501


def suppress_plain_keyword(keyword: str, prompt: str):
    return re.sub(
        re.compile(RE_PLAIN_KEYWORD.format(keyword=keyword), re.I), r"\2", prompt
    )


def suppress_lora_keyword(keyword: str, prompt: str):
    return re.sub(
        re.compile(RE_LORA_KEYWORD.format(keyword=keyword), re.I), r"\2", prompt
    )
