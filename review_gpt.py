from datasets import load_dataset

saved_data = load_dataset("UofA-LINGO/wikipedia-summaries-gpt-triples")
choice = input("Use drop file? ")
if choice in ['y', 'Y', 'yes', 'Yes']:
    inds = []
    rev = []
    drop = []
    with open("gpt-rev.txt") as f:
        inds = f.readlines()
    for i in inds:
        if i.strip().isnumeric():
            print(i)
            print(saved_data['train']["Summaries"][int(i)])
            review = input("Mark review? ")
            if review in ['y', 'Y', 'yes', 'Yes']:
                rev.append(int(i))
            else:
                drop.append(int(i))
    up = input("Update? ")
    if up in ['y', 'Y', 'yes', 'Yes']:
        with open("gpt-rev.txt", 'w') as f:
            print("Review \n\n", file=f)
            for r in rev:
                print(r, file=f)
            print("\n\nRemove\n\n", file=f)
            for d in drop:
                print(d, file=f)
else:
    """
    while(True):
        ind = int(input("INDEX: "))
        print(saved_data['train']["Summaries"][ind])
    """
    start = int(input("Starting index: "))
    dropList = []
    for i in range(start, len(saved_data['train']['text'])):
        print(i)
        print(saved_data['train']['text'][i] + '\n')
        print(saved_data['train']['triples'][i] + '\n\n')
        drop = input("Mark to review?: ")
        if drop in ['y', 'Y', 'yes', 'Yes']:
            dropList.append(i)
        elif drop in ['exit', 'Exit']:
            break
    with open("gpt-rev.txt", 'a') as f:
        for d in dropList:
            print(d, file=f)