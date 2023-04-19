import re
import tiktoken

enc = tiktoken.get_encoding("gpt2")

BY_CASE = re.compile(r' ?([^a-z ][^A-Z ]+|[^A-Z ]+|[^a-z ]+| +)')
def split_by_case(s):
    results = []
    while s:
        m = BY_CASE.match(s)
        assert m
        results.append(m.group())
        ss = s[m.end():]
        assert ss != s, (s, ss)
        s = ss
    return results
assert split_by_case("hello") == ["hello"]
assert split_by_case("HELLO") == ["HELLO"]
assert split_by_case("Hello") == ["Hello"]
assert split_by_case(" hello") == [" hello"]
assert split_by_case(" Hello") == [" Hello"]
assert split_by_case("helloWorld") == ["hello", "World"]
assert split_by_case("HELLo") == ["HELL", "o"]
assert split_by_case(" snakeCase_is_snaky and also spaces exist") == [" snake", "Case_is_snaky", " and", " also", " spaces", " exist"]
assert split_by_case("First Citizen:") == ["First", " Citizen:"]
assert split_by_case(" Citizen") == [" Citizen"]
assert split_by_case("CitiZEN") == ["Citi", "ZEN"]
assert split_by_case("CitiZEn") == ["Citi", "ZE", "n"]

def add_flags(s):
    flags = [False] * 4
    flags[0] = s.islower()
    flags[1] = s.isupper()
    flags[2] = s.istitle()
    flags[3] = s.startswith(" ")
    if not s[0].isspace():
        s = " " + s
    return s.lower(), flags
assert add_flags("Citizen") == (" citizen", [False, False, True, False])
assert add_flags(" Citizen") == (" citizen", [False, False, True, True])
assert add_flags("citizen") == (" citizen", [True, False, False, False])
assert add_flags(" citizen") == (" citizen", [True, False, False, True])

def apply_flags(s, flags):
    if flags[1]:
        s = s.upper()
    if flags[2]:
        s = s.title()
    if not flags[3] and s.startswith(" "):
        s = s[1:]
    return s
assert apply_flags(*add_flags("mom")) == "mom"
assert apply_flags(*add_flags(" mom")) == " mom"
assert apply_flags(*add_flags("  mom")) == "  mom"
assert apply_flags(*add_flags("   mom")) == "   mom"

def encode(s):
    lowercase_tokens = []
    flags = []
    starts_with_nonspace = s and not s[0].isspace()
    if starts_with_nonspace:
        s = " " + s
    for token in enc.encode_ordinary(s):
        case_sensitive = enc.decode([token])
        for s in split_by_case(case_sensitive):
            s, s_flags = add_flags(s)
            tokens = enc.encode_ordinary(s)
            lowercase_tokens.append(tokens[0])
            flags.append(s_flags)
            for token in tokens[1:]:
                lowercase_tokens.append(token)
                flags.append(add_flags("HELLO" if s_flags[1] else "hello")[1])
    if starts_with_nonspace:
        assert flags[0][3]
        flags[0][3] = False
    return lowercase_tokens, flags

def decode(idx, flags):
    results = []
    for token, token_flags in zip(idx, flags):
        s = enc.decode([token])
        s = apply_flags(s, token_flags)
        results.append(s)
    return ''.join(results)

def roundtrip(s):
    return decode(*encode(s))

def check_roundtrip(s):
    assert roundtrip(s) == s, (s, encode(s), roundtrip(s))
check_roundtrip("hello world")
check_roundtrip("Hello World")
check_roundtrip("Hello, World!")
check_roundtrip("hi  mom")
check_roundtrip(" hi mom")
check_roundtrip("hi mom ")
check_roundtrip(" hi mom ")
check_roundtrip("  hi")
check_roundtrip("    Hello,    World !   ")
check_roundtrip("""\
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?""")
check_roundtrip("HORTENSIO")
check_roundtrip("HORTENSIO:")
check_roundtrip("\nHORTENSIO:\n")
check_roundtrip("\n\nHORTENSIO:\n")
check_roundtrip("How now, my friend! why dost thou look so pale?\n\nHORTENSIO:\nFor fear, I promise you, if I look pale.")
assert len(encode(" Citizen")[0]) == 1
assert len(encode(" citizen")[0]) == 1
assert len(encode("Citizen")[0]) == 1
assert len(encode("citizen")[0]) == 1
assert encode("citiZEN") == (
    [269, 340, 72, 1976, 551],
    [
        [True, False, False, False], 
        [True, False, False, False], 
        [True, False, False, False], 
        [False, True, True, False], 
        [False, True, False, False]
    ]
)
assert len(encode("citiZEN")[0]) == 5
assert len(encode("CITIzen")[0]) == 5
assert len(encode("hear")[0]) == 1
assert len(encode("First Citizen")[0]) == 2
