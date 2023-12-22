import UnlimitedGPT as gpt

print("File left for possible use in future, although OpenAI does not like the use of unoffical APIs, so do not use this over a long period of time.")
session_key = ""
conversation_id_gpt4 = 'f7b4b745-5999-432f-9678-069d5208fe81'  # GPT-4 conversation ID
conversation_id_gpt3 = '869950cd-b272-49b6-aa2a-ace81c17b3ba'  # GPT-3.5 conversation ID
print("Initializing API.")
#api = gpt.ChatGPT(session_key, conversation_id=conversation_id_gpt3)
print("API ready.")
input_text = "Pretend that you are great at parsing a text and can effectively extract all entities and their relationships from a text. I will give you a text and you extract all the possible triples. First, extract the triples in textual format, then based on the text, link them to DBPedia vocabularies. Please extract all the triples using the DBpedia vocabulary. Instead of using dbpedia:*, please use dbr:*, dbo:*, dbp:*, or dbc:*. \
The output format is as below Text-Link Triples: \
(subject | relationship | object) \
DBpedia-Linked Triples: \
(subject | relationship | object) \
Text: The Haunting of Castle Malloy is the 19th installment in the Nancy Drew point-and-click adventure game series by Her Interactive. The game is available for play on Microsoft Windows platforms. It has an ESRB rating of E for moments of mild violence and peril. Players take on the first-person view of fictional amateur sleuth Nancy Drew and must solve the mystery through interrogation of suspects, solving puzzles, and discovering clues. There are two levels of gameplay, Junior and Senior detective modes, each offering a different difficulty level of puzzles and hints, however neither of these changes affects the actual plot of the game. The game is loosely based on a book entitled The Bike Tour Mystery (2002).[1][2] \
\
Plot[edit] \
Nancy Drew travels to Castle Malloy in Ireland to be the maid of honor in her friend Kyler Mallory's wedding. As Nancy drives towards the castle, a ghostly figure darts out in front of her car. The car crashes into a ditch, and Nancy is stuck at the site of a new mystery. Kyler's wedding was supposed to be the biggest event in Bailor since half the castle blew up sixty years ago, but now the groom is missing. Was he snatched by the banshee rumored to haunt the castle, or is this a case of cold feet? It's up to Nancy to find the groom and save this wedding. \
Development[edit] \
Characters[edit] \
Nancy Drew - Nancy is an 18-year-old amateur detective from the fictional town of River Heights in the United States. She is the only playable character in the game, which means the player must solve the mystery from her perspective. \
Kyler Mallory - Kyler is a stockbroker from London who lived with Nancy\'s family as an exchange student four years ago. She is now engaged to Matt Simmons and is excited to hold her wedding at Castle Malloy. Her grandfather, Edmund Mallory, recently died and named her as his sole beneficiary. He never spoke about his childhood and Kyler is just now learning about her true family history. She is certain that Matt is playing a prank. \
Matt Simmons - Matt is a travel and sailing freelance writer for magazines. Known for his dry wit, he loves to play pranks on others. He\'s marrying Kyler and wants to make the occasion something no one will ever forget, but has he gone too far this time? \
Kit Foley - Matt\'s best friend and accomplice on sailing trips. Kit was born in the US, but his family moved to London when Kit was young. He was never able to pick up the British accent. He currently works as a land developer, and he arrived early to help Matt and Kyler prepare for their wedding. He believes that Matt got cold-feet and ran away, but he won\'t admit that to Kyler. \
Donal Delany - Castle Malloy\'s ever-present caretaker, Donal has been overseeing the grounds for too many years to count. This Irish-born gentleman isn\'t particularly fond of foreigners, and he\'s the first to admit it. He is the resident expert on Irish mythology and lore, which might explain his superstitious nature. He believes that Matt was kidnapped by fairies. \
Cast[edit] \
Nancy Drew - Lani Minella \
Donal Delany - Evan Mosher \
Kyler Mallory - Alycia Delmore \
Kit Foley - Matthew C. Shimkus \
Matt Simmons - Peter Dylan O\'Connor \
Alan Paine - Daniel Christenson \
Ned Nickerson - Scott Carty \
Bess Marvin - Jennifer Pratt \
George Fayne - Patty Pomplun \
Seamus - Jonah Von Spreekin [3]"
# message = api.send_message(
#     "Please extract triples from the following text, providing output in the form of (subject, predicate, object): The three laws of motion were first stated by Isaac Newton in his Philosophiæ Naturalis Principia Mathematica (Mathematical Principles of Natural Philosophy), originally published in 1687.",
#     input_mode="SLOW",
#     input_delay=0.1,
#     continue_generating=True
# )
#print(message.response)