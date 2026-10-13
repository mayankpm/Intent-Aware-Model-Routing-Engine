"""Write research/data/custom.jsonl: 60 hand-written, automatically checkable items.

Three task types have no small, auto-graded public benchmark that fits this
project, so they are written here: instruction following (20), specialized
NLP tasks (20) and business-rule reasoning (20). Each item carries a reference
response that must pass its own checker (validated in tests/test_research.py),
and most numeric answers note the arithmetic behind them in a `work` field.

Usage: python -m research.build_custom
"""
import json
from pathlib import Path

NUM = "\nGive the final answer on the last line as 'Answer: <number>'."


def lab(options):
    return "\nEnd your reply with 'Answer: " + " or ".join(options) + "'."


ITEMS = [
    # ---------------------------------------------------------------- instruction following
    dict(id="if-01", task="instruction", prompt="Write a poem about the ocean in exactly 4 lines.",
         check={"type": "constraints", "lines": 4},
         reference="Waves roll in under a silver sky\nSalt wind carries the gulls up high\nDeep blue water keeps its ancient song\nThe tide returns where it belongs"),
    dict(id="if-02", task="instruction", prompt="Write a haiku about snow. Use exactly 3 lines and nothing else.",
         check={"type": "constraints", "lines": 3},
         reference="Soft snow on the roof\nThe quiet street turns to white\nFootprints fade by dawn"),
    dict(id="if-03", task="instruction", prompt="Write a limerick about a frog. It must have exactly 5 lines.",
         check={"type": "constraints", "lines": 5},
         reference="A frog with a very loud croak\nSat down on a lily of oak\nHe sang all the night\nTill the moon lost its light\nAnd the pond-folk all said it's no joke"),
    dict(id="if-04", task="instruction", prompt="Write a poem about the city in exactly 4 lines, where every line starts with the letter S.",
         check={"type": "constraints", "lines": 4, "lines_start": ["S", "S", "S", "S"]},
         reference="Streets hum with taxis late at night\nSkyscrapers glow with borrowed light\nStrangers pass with hurried feet\nSirens sing along the street"),
    dict(id="if-05", task="instruction", prompt="Write a story about a robot in under 80 words that includes the words 'battery', 'garden' and 'moon', and does not use the word 'the'.",
         check={"type": "constraints", "max_words": 79, "include": ["battery", "garden", "moon"], "exclude": ["the"]},
         reference="A small robot named Pip tended a garden each night. Its battery ran low, so it rested under a bright moon and dreamed of tomatoes."),
    dict(id="if-06", task="instruction", prompt="Write exactly 3 lines about winter. The first line must start with 'Cold', and the word 'fire' must appear in the last line.",
         check={"type": "constraints", "lines": 3, "lines_start": ["Cold", None, None], "line_contains": {"-1": ["fire"]}},
         reference="Cold winds rattle every pane\nSnow lies heavy on the lane\nWe gather close beside the fire"),
    dict(id="if-07", task="instruction", prompt="Write a 4-line poem about coffee in which no line has more than 6 words.",
         check={"type": "constraints", "lines": 4, "max_words_per_line": 6},
         reference="Morning steam curls from my cup\nDark and bitter, warm enough\nThe world wakes slowly here\nOne sip and skies clear"),
    dict(id="if-08", task="instruction", prompt="Write a story in exactly 5 numbered sentences (1. to 5.) about a lost key.",
         check={"type": "constraints", "numbered_items": 5, "include": ["key"]},
         reference="1. Mia lost her key on Monday.\n2. She searched the kitchen twice.\n3. Her dog barked at the sofa.\n4. Under a cushion lay the key.\n5. She gave the dog a treat."),
    dict(id="if-09", task="instruction", prompt="Write an acrostic poem for the word 'RIVER': 5 lines whose first letters spell R, I, V, E, R.",
         check={"type": "constraints", "lines": 5, "lines_start": ["R", "I", "V", "E", "R"]},
         reference="Rushing past the stones\nIn the morning light\nVeins of silver water\nEndless in their flight\nRunning to the sea"),
    dict(id="if-10", task="instruction", prompt="Write a description of a market in under 50 words that mentions the colors red, blue and green and no other color words.",
         check={"type": "constraints", "max_words": 49, "include": ["red", "blue", "green"],
                "exclude": ["yellow", "orange", "purple", "pink", "black", "white", "brown", "gray", "grey", "gold", "silver", "violet"]},
         reference="Stalls overflow with red peppers, blue scarves and green herbs while vendors call out prices and children weave between the crowded tables."),
    dict(id="if-11", task="instruction", prompt="Hey, my name is Priya. Please greet me using my name in one short sentence.",
         check={"type": "constraints", "include": ["Priya"], "max_words": 20},
         reference="Hello Priya, it's lovely to meet you!"),
    dict(id="if-12", task="instruction", prompt="Reply with only the word 'yes'.",
         check={"type": "constraints", "equals": "yes"},
         reference="Yes"),
    dict(id="if-13", task="instruction", prompt="I'm planning a weekend. Suggest exactly three activities as a numbered list, and nothing else.",
         check={"type": "constraints", "lines": 3, "numbered_items": 3},
         reference="1. Go for a hike\n2. Visit a museum\n3. Cook a new recipe"),
    dict(id="if-14", task="instruction", prompt="My name is Daniel and I live in Oslo. Greet me by name, mention my city, and ask me exactly one question.",
         check={"type": "constraints", "include": ["Daniel", "Oslo"], "question_marks": 1},
         reference="Hi Daniel, greetings to you in Oslo! What do you enjoy most about living there?"),
    dict(id="if-15", task="instruction", prompt="Recommend two books as a bulleted list using '-' at the start of each line, with no other text.",
         check={"type": "constraints", "lines": 2, "lines_start": ["-", "-"]},
         reference="- The Hobbit by J.R.R. Tolkien\n- Dune by Frank Herbert"),
    dict(id="if-16", task="instruction", prompt="Reply in all lowercase letters: tell me one fun fact about cats.",
         check={"type": "constraints", "lowercase": True, "include_any": [["cat", "cats"]]},
         reference="cats spend about two thirds of their lives sleeping."),
    dict(id="if-17", task="instruction", prompt="I have a job interview tomorrow. Give me exactly 3 tips, each on its own line starting with 'Tip:', and nothing else.",
         check={"type": "constraints", "lines": 3, "lines_start": ["Tip:", "Tip:", "Tip:"]},
         reference="Tip: Research the company.\nTip: Prepare examples of your work.\nTip: Get a good night's sleep."),
    dict(id="if-18", task="instruction", prompt="Tell me a joke about computers in under 25 words, and end your reply with the word 'lol'.",
         check={"type": "constraints", "max_words": 24, "ends_with": "lol"},
         reference="Why did the computer go to the doctor? It had a virus, lol"),
    dict(id="if-19", task="instruction", prompt="Answer these two questions on two separate lines and nothing else: what is 3 times 4, and what is the capital of Japan?",
         check={"type": "constraints", "lines": 2, "line_contains": {"0": ["12"], "1": ["Tokyo"]}},
         reference="12\nTokyo"),
    dict(id="if-20", task="instruction", prompt="Respond with a JSON object with keys 'mood' and 'reply', and nothing else.",
         check={"type": "json_keys", "keys": ["mood", "reply"]},
         reference='{"mood": "cheerful", "reply": "Happy to help!"}'),

    # ---------------------------------------------------------------- specialized NLP tasks
    dict(id="sp-01", task="specialized", prompt="Classify the sentiment of this review as positive or negative: 'I loved this phone, the battery lasts for days.'" + lab(["positive", "negative"]),
         check={"type": "label", "options": ["positive", "negative"], "answer": "positive"}, reference="Answer: positive"),
    dict(id="sp-02", task="specialized", prompt="Classify the sentiment of this review as positive or negative: 'The food was cold and the waiter was rude.'" + lab(["positive", "negative"]),
         check={"type": "label", "options": ["positive", "negative"], "answer": "negative"}, reference="Answer: negative"),
    dict(id="sp-03", task="specialized", prompt="Classify the sentiment (positive, negative or neutral) of this sentence: 'The meeting is scheduled for 3 pm on Tuesday.'" + lab(["positive", "negative", "neutral"]),
         check={"type": "label", "options": ["positive", "negative", "neutral"], "answer": "neutral"}, reference="Answer: neutral"),
    dict(id="sp-04", task="specialized", prompt="Translate into Spanish: 'The cat is black.'",
         check={"type": "constraints", "include": ["gato", "negro"]}, reference="El gato es negro."),
    dict(id="sp-05", task="specialized", prompt="Translate into French: 'Thank you very much.'",
         check={"type": "constraints", "include": ["merci beaucoup"]}, reference="Merci beaucoup."),
    dict(id="sp-06", task="specialized", prompt="Translate into German: 'Good morning.'",
         check={"type": "constraints", "include": ["guten morgen"]}, reference="Guten Morgen."),
    dict(id="sp-07", task="specialized", prompt="Translate into French: 'The children are playing in the garden.'",
         check={"type": "constraints", "include": ["enfants", "jardin"]}, reference="Les enfants jouent dans le jardin."),
    dict(id="sp-08", task="specialized", prompt="Translate into Spanish: 'I would like a table for two people, please.'",
         check={"type": "constraints", "include": ["mesa", "dos", "por favor"]}, reference="Quisiera una mesa para dos personas, por favor."),
    dict(id="sp-09", task="specialized", prompt="Extract the person's name from this sentence and reply with only the name: 'Yesterday, Maria visited the museum.'",
         check={"type": "constraints", "include": ["Maria"], "max_words": 3}, reference="Maria"),
    dict(id="sp-10", task="specialized", prompt="Extract all the people's names from this text and list them separated by commas, in the order they appear: 'At the meeting, Ahmed presented the budget, Lena asked about hiring, and Carlos closed the session.'",
         check={"type": "ordered", "items": ["Ahmed", "Lena", "Carlos"]}, reference="Ahmed, Lena, Carlos"),
    dict(id="sp-11", task="specialized", prompt="Extract every date from this text and list them in the format YYYY-MM-DD, one per line: 'The contract was signed on March 3, 2021, renewed on 15 January 2023, and ends on 2025-06-30.'",
         check={"type": "constraints", "include": ["2021-03-03", "2023-01-15", "2025-06-30"]}, reference="2021-03-03\n2023-01-15\n2025-06-30"),
    dict(id="sp-12", task="specialized", prompt="Extract the email addresses from this text, one per line: 'Contact sales@acme.com for pricing or support@acme.org for help. Ignore www.acme.com.'",
         check={"type": "constraints", "include": ["sales@acme.com", "support@acme.org"], "exclude": ["www.acme.com"]}, reference="sales@acme.com\nsupport@acme.org"),
    dict(id="sp-13", task="specialized", prompt="Summarize in under 15 words: 'The city council voted to build a new public library downtown, which will open next year.'",
         check={"type": "constraints", "max_words": 14, "include": ["library"]}, reference="Council approves new downtown public library opening next year."),
    dict(id="sp-14", task="specialized", prompt="Summarize in at most 20 words, and include the word 'solar': 'Solar panels convert sunlight into electricity. Prices have dropped by more than 80 percent over the past decade, and many homes and businesses now install them on their roofs.'",
         check={"type": "constraints", "max_words": 20, "include": ["solar"]}, reference="Solar panel prices fell over 80 percent in a decade, so many homes and businesses now install them."),
    dict(id="sp-15", task="specialized", prompt="Classify each review as positive or negative, one label per line, in order: 1) 'Great service and fast delivery.' 2) 'The product broke after one day.' 3) 'Absolutely worth the price.'",
         check={"type": "label_sequence", "options": ["positive", "negative"], "answer": ["positive", "negative", "positive"]}, reference="positive\nnegative\npositive"),
    dict(id="sp-16", task="specialized", prompt="Convert this into a JSON array of objects with keys 'name' and 'age': Alice is 30, Bob is 25. Return only JSON.",
         check={"type": "json_equals", "answer": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}, reference='[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]'),
    dict(id="sp-17", task="specialized", prompt="Correct the spelling in this sentence and return only the corrected sentence: 'I recieved the pakage yesterday.'",
         check={"type": "constraints", "include": ["received", "package"], "exclude": ["recieved", "pakage"]}, reference="I received the package yesterday."),
    dict(id="sp-18", task="specialized", prompt="What language is this sentence in: 'Gracias por tu ayuda'? Answer with one word.",
         check={"type": "constraints", "include": ["Spanish"], "max_words": 3}, reference="Spanish"),
    dict(id="sp-19", task="specialized", prompt="Classify the topic of this headline as SPORTS, POLITICS or TECHNOLOGY: 'New smartphone chip doubles battery life.'" + lab(["SPORTS", "POLITICS", "TECHNOLOGY"]),
         check={"type": "label", "options": ["SPORTS", "POLITICS", "TECHNOLOGY"], "answer": "TECHNOLOGY"}, reference="Answer: TECHNOLOGY"),
    dict(id="sp-20", task="specialized", prompt="Count the number of words in this sentence: 'The quick brown fox jumps over the lazy dog.'" + NUM,
         check={"type": "number", "answer": 9}, reference="Answer: 9"),

    # ---------------------------------------------------------------- business rules
    dict(id="br-01", task="business_rules", prompt="Policy: orders over 50 dollars ship free; otherwise shipping costs 5 dollars. An order totals 62 dollars. What is the shipping cost in dollars?" + NUM,
         check={"type": "number", "answer": 0}, reference="Answer: 0", work="62 > 50 -> free"),
    dict(id="br-02", task="business_rules", prompt="Policy: refunds are allowed within 30 days of purchase. A customer bought an item 12 days ago. Should the refund be approved?" + lab(["APPROVE", "DENY"]),
         check={"type": "label", "options": ["APPROVE", "DENY"], "answer": "APPROVE"}, reference="Answer: APPROVE", work="12 <= 30"),
    dict(id="br-03", task="business_rules", prompt="Policy: refunds are allowed within 30 days of purchase. A customer bought an item 45 days ago. Should the refund be approved?" + lab(["APPROVE", "DENY"]),
         check={"type": "label", "options": ["APPROVE", "DENY"], "answer": "DENY"}, reference="Answer: DENY", work="45 > 30"),
    dict(id="br-04", task="business_rules", prompt="Policy: employees get 1.5 times their hourly rate for hours over 40 in a week. Ravi earns 20 dollars an hour and worked 40 hours. What is his weekly pay in dollars?" + NUM,
         check={"type": "number", "answer": 800}, reference="Answer: 800", work="40*20"),
    dict(id="br-05", task="business_rules", prompt="Rule: a password is valid only if it has at least 8 characters. Is the password 'sunflower' valid?" + lab(["VALID", "INVALID"]),
         check={"type": "label", "options": ["VALID", "INVALID"], "answer": "VALID"}, reference="Answer: VALID", work="len('sunflower') = 9"),
    dict(id="br-06", task="business_rules", prompt="Rule: customers with more than 1000 loyalty points get Gold status; otherwise Silver. A customer has 850 points. Which status?" + lab(["GOLD", "SILVER"]),
         check={"type": "label", "options": ["GOLD", "SILVER"], "answer": "SILVER"}, reference="Answer: SILVER", work="850 <= 1000"),
    dict(id="br-07", task="business_rules", prompt="Policy: a support ticket is urgent only if the customer is on the Enterprise plan. A Basic plan customer opens a ticket. Is it urgent?" + lab(["URGENT", "NORMAL"]),
         check={"type": "label", "options": ["URGENT", "NORMAL"], "answer": "NORMAL"}, reference="Answer: NORMAL"),
    dict(id="br-08", task="business_rules", prompt="Pricing: each seat costs 12 dollars per month. A team buys 10 seats. What is the monthly cost in dollars?" + NUM,
         check={"type": "number", "answer": 120}, reference="Answer: 120", work="12*10"),
    dict(id="br-09", task="business_rules", prompt="Rule: an invoice is OVERDUE if it is still unpaid more than 30 days after its due date; otherwise it is CURRENT. An invoice is unpaid 31 days after its due date. What is its status?" + lab(["OVERDUE", "CURRENT"]),
         check={"type": "label", "options": ["OVERDUE", "CURRENT"], "answer": "OVERDUE"}, reference="Answer: OVERDUE"),
    dict(id="br-10", task="business_rules", prompt="Policy: a discount code gives 10% off orders of 100 dollars or more. An order is 100 dollars. What is the price after the discount, in dollars?" + NUM,
         check={"type": "number", "answer": 90}, reference="Answer: 90", work="100*0.9"),
    dict(id="br-11", task="business_rules", prompt="Shipping policy: orders under 50 dollars pay 8 dollars shipping. Orders from 50 to 99.99 dollars pay 4 dollars. Orders of 100 dollars or more ship free. Members get an extra 2 dollars off shipping, but shipping can never be negative. A member orders 75 dollars of goods. What is the total charge (goods plus shipping) in dollars?" + NUM,
         check={"type": "number", "answer": 77}, reference="Answer: 77", work="75 + max(0, 4-2)"),
    dict(id="br-12", task="business_rules", prompt="Refund policy: unopened items can be refunded within 30 days. Opened items can be refunded within 14 days, but only if defective. A customer bought an item 10 days ago, opened it, and it is not defective. Should the refund be approved?" + lab(["APPROVE", "DENY"]),
         check={"type": "label", "options": ["APPROVE", "DENY"], "answer": "DENY"}, reference="Answer: DENY"),
    dict(id="br-13", task="business_rules", prompt="Overtime policy: hours over 40 in a week are paid at 1.5 times the hourly rate, and hours over 50 are paid at 2 times the hourly rate. Lina earns 20 dollars an hour and worked 55 hours. What is her weekly pay in dollars?" + NUM,
         check={"type": "number", "answer": 1300}, reference="Answer: 1300", work="40*20 + 10*30 + 5*40"),
    dict(id="br-14", task="business_rules", prompt="Discount rules: a 20% member discount is applied first, then a 10 dollar coupon, but the coupon only applies if the price after the member discount is at least 50 dollars. A member buys an item priced 60 dollars with the coupon. What do they pay in dollars?" + NUM,
         check={"type": "number", "answer": 48}, reference="Answer: 48", work="60*0.8 = 48 < 50, no coupon"),
    dict(id="br-15", task="business_rules", prompt="Loan rule: approve if the credit score is at least 700 and the debt-to-income ratio is below 0.35; if the score is between 650 and 699, approve only if the ratio is below 0.2; otherwise deny. An applicant has a score of 680 and a ratio of 0.25." + lab(["APPROVE", "DENY"]),
         check={"type": "label", "options": ["APPROVE", "DENY"], "answer": "DENY"}, reference="Answer: DENY", work="680 in 650-699, 0.25 >= 0.2"),
    dict(id="br-16", task="business_rules", prompt="Support routing: tickets mentioning 'outage' go to Tier 3. Otherwise, Enterprise customers go to Tier 2 and everyone else to Tier 1. An Enterprise customer writes: 'We have an outage in our EU region.' Which tier number?" + NUM,
         check={"type": "number", "answer": 3}, reference="Answer: 3"),
    dict(id="br-17", task="business_rules", prompt="Inventory rule: reorder when stock falls below the reorder point, which is daily demand times lead time in days plus 20 units of safety stock. Daily demand is 15 units and lead time is 4 days. Current stock is 75 units. Should we reorder?" + lab(["REORDER", "HOLD"]),
         check={"type": "label", "options": ["REORDER", "HOLD"], "answer": "REORDER"}, reference="Answer: REORDER", work="15*4+20 = 80 > 75"),
    dict(id="br-18", task="business_rules", prompt="Pricing: a plan costs 30 dollars per user per month for the first 10 users and 25 dollars per user per month for each user after that. Annual billing charges for 10 months instead of 12. What is the annual price for 14 users on annual billing, in dollars?" + NUM,
         check={"type": "number", "answer": 4000}, reference="Answer: 4000", work="(10*30 + 4*25) * 10"),
    dict(id="br-19", task="business_rules", prompt="Leave policy: employees earn 1.5 vacation days per month worked. An employee starts the year with 5 days, works all 12 months, and takes 4 days off at the end of the year. What is the balance at year end, in days?" + NUM,
         check={"type": "number", "answer": 19}, reference="Answer: 19", work="5 + 1.5*12 - 4"),
    dict(id="br-20", task="business_rules", prompt="Escalation rule: a complaint is escalated if the customer has complained at least 3 times in 30 days, or if the order value is over 500 dollars and the customer is a VIP. A VIP customer with an order of 450 dollars has complained 2 times this month." + lab(["ESCALATE", "STANDARD"]),
         check={"type": "label", "options": ["ESCALATE", "STANDARD"], "answer": "STANDARD"}, reference="Answer: STANDARD", work="2 < 3 and 450 <= 500"),
]


def main():
    out = Path(__file__).parent / "data" / "custom.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for it in ITEMS:
            it = {"source": "custom", "max_tokens": 256, **it}
            f.write(json.dumps(it) + "\n")
    print(f"wrote {len(ITEMS)} items to {out}")


if __name__ == "__main__":
    main()
