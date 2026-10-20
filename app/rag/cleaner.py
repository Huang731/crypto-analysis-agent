"""入库前的固定文本清洗，保留 Markdown 结构和链接出处。"""

from __future__ import annotations

import re


class CleanProcessor:
    """统一清理 MD、TXT 和 PDF 元素正文，不依赖规则配置。"""

    @staticmethod
    def clean(text: str) -> str:
        """先清理无效字符，再对代码块以外的正文执行固定清洗。"""
        # 先统一换行，避免 Windows 回车被后续空白压缩误处理。
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("<|", "<").replace("|>", ">")
        # 保留制表符和换行；其余指定控制字符及 U+FFFE 按原逻辑删除。
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffe]", "", text)

        parts: list[str] = []
        prose: list[str] = []
        fence_char = ""
        fence_length = 0
        indented_code = False
        for line in text.splitlines(keepends=True):
            if fence_char:
                # 围栏内保留空白、URL 和邮箱，防止改写代码示例。
                parts.append(line)
                if re.fullmatch(
                    rf" {{0,3}}{re.escape(fence_char)}{{{fence_length},}}[ \t]*\n?",
                    line,
                ):
                    fence_char = ""
                continue

            # 缩进代码及其间的空行整体保留，避免空行压缩改变代码布局。
            if line.startswith(("    ", "\t")) or (indented_code and not line.strip()):
                if prose:
                    parts.append(CleanProcessor._clean_prose("".join(prose)))
                    prose.clear()
                parts.append(line)
                indented_code = True
                continue
            indented_code = False

            fence = re.match(r" {0,3}(`{3,}|~{3,})([^\n]*)", line)
            # 反引号围栏的信息字符串不能再包含反引号。
            if fence and not (fence.group(1).startswith("`") and "`" in fence.group(2)):
                if prose:
                    parts.append(CleanProcessor._clean_prose("".join(prose)))
                    prose.clear()
                parts.append(line)
                fence_char = fence.group(1)[0]
                fence_length = len(fence.group(1))
            else:
                prose.append(line)

        # 未闭合的代码围栏也保留到文末；只清洗已经收集的普通正文。
        if prose:
            parts.append(CleanProcessor._clean_prose("".join(prose)))
        return "".join(parts)

    @staticmethod
    def _clean_prose(text: str) -> str:
        """逐行清洗正文，仅在正文中把连续三个以上换行压缩成两个。"""
        lines = [
            CleanProcessor._clean_line(line) for line in text.splitlines(keepends=True)
        ]
        return re.sub(r"\n{3,}", "\n\n", "".join(lines))

    @staticmethod
    def _clean_line(line: str) -> str:
        """保留行首缩进、行尾硬换行及内联结构，只处理结构之间的正文。"""
        ending = "\n" if line.endswith("\n") else ""
        content = line.removesuffix("\n")
        indent_length = len(content) - len(content.lstrip(" \t"))
        indent, body = content[:indent_length], content[indent_length:]

        # Markdown 用行尾至少两个空格表达硬换行，不能压成一个空格。
        hard_break = ""
        if body.endswith("  "):
            body_without_spaces = body.rstrip(" ")
            hard_break = body[len(body_without_spaces) :]
            body = body_without_spaces

        # 链接和图片共用匹配，直接保留整个原文，避免捕获组错误和占位符冲突。
        # URL 支持一层嵌套括号；内联代码及 HTML 标签也保留，防止破坏表格属性。
        protected = re.compile(
            r"(?P<ticks>`+)(?!`)[^\n]*?(?<!`)(?P=ticks)(?!`)"
            r"|!?\[[^\]\n]*\]\(https?://(?:\\.|[^()\\\n]|\([^()\n]*\))*\)"
            r"|</?[A-Za-z][A-Za-z0-9:-]*(?:[ \t]+[^<>\n]*)?[ \t]*/?>"
        )
        parts = [indent]
        start = 0
        for match in protected.finditer(body):
            parts.append(CleanProcessor._clean_plain(body[start : match.start()]))
            parts.append(match.group())
            start = match.end()
        parts.append(CleanProcessor._clean_plain(body[start:]))
        parts.extend((hard_break, ending))
        return "".join(parts)

    @staticmethod
    def _clean_plain(text: str) -> str:
        """删除未受保护的 URL、邮箱，并压缩连续的横向空白。"""
        text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "", text)
        # 中文句末标点不属于 URL，保留它们以免把后面的正文一并删除。
        text = re.sub(r"https?://[^\s<>\"'，。！？；：、（）【】《》]+", "", text)
        return re.sub(
            r"[\t\x20\u00a0\u1680\u180e\u2000-\u200a\u202f\u205f\u3000]{2,}",
            " ",
            text,
        )
