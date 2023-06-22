def CheckASCII(text):
    # Allow ASCII characters as well as Greek alphabet
    for c in text:
        if ord(c) > 127 and not (ord(c) >= 880 and ord(c) <= 1023): 
            return False
    return True

def stoa(text):
    return [ord(c) for c in text]

with open("../dbo.xml", encoding='utf8') as f:
    # [0] is XML header info, [1] is RDF declaration
    lines = f.readlines()
    startTags = lines[:2]
    lines = lines[2:]
    endTag = lines.pop()

cleaned = startTags

for i in range(0, len(lines), 3):
    openTag = lines[i]
    closeTag = lines[i+2]
    if openTag.find("rdf:Description") == -1:
        print(f"Offset at {i}")
        print(openTag)
        print(lines[i+1])
        print(closeTag)
        exit(-1)
    elif closeTag.find("rdf:Description") == -1:
        print(f"Offset at {i}")
        print(openTag)
        print(lines[i+1])
        print(closeTag)
        exit(-1)

# Lines are all in groupings of three, opening description, about section, and closing tag.
for i in range(0, len(lines), 3):
    openTag = lines[i]
    about = lines[i+1]

    # Remove non-ASCII chars
    if False == CheckASCII(openTag):
        continue

    if about.find("xml:lang=") != -1:
        ind = about.find("xml:lang=")+len("xml:lang=")
        if about[ind:ind+4] != "\"en\"":
            continue
    if False == CheckASCII(about):
        continue
    
    cleaned.append(openTag)
    cleaned.append(about)
    cleaned.append(lines[i+2])
cleaned.append(endTag)

print(f"English only ontology is {len(cleaned)} lines.")
with open("../dbo_en.xml", 'w', encoding='utf8') as f:
    for i in range(len(cleaned)):
        try:
            print(cleaned[i], file=f, end="")
        except UnicodeEncodeError:
            print(f"UnicodeEncodeError on {i}")
            print(cleaned[i])
            pass