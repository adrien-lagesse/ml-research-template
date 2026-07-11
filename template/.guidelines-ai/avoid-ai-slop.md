# Avoid AI-slop writing

A short field guide for keeping prose human. Applies to docs, comments, commit
messages, PR descriptions, and chat replies.


> These are signals, not proof. Technical genres and non-native writing
> legitimately produce similar shapes. Use as indicators, not verdicts.

## 5 core principles

1. **Vary sentence length**: mix short (3–8 words) with long (20+). Fragments are fine.
2. **Be concrete**: replace vague claims with numbers, names, dates, examples.
3. **Have a voice**: use first person where it fits; state preferences; show reactions.
4. **Cut neutrality**: take a position instead of hedging.
5. **Earn emphasis**: don't say something is interesting; make it interesting.

## Banned words (replace with plain alternatives)

delve → look at · leverage → use · utilize → use · robust → solid ·
comprehensive → thorough · seamless → smooth · meticulous → careful ·
pivotal/crucial → key · underscore → highlight · harness → use ·
facilitate → help · foster → support · streamline → simplify ·
elevate → improve · realm/landscape → field · cutting-edge → latest ·
testament to → shows · in order to → to · due to the fact that → because ·
serves as / boasts / features → is / has · knob → setting / parameter

Metaphor words to cut entirely: tapestry, beacon, symphony, ecosystem,
nestled, vibrant, thriving, bustling, ever-evolving.

## Banned phrase templates

- "It's not just X, it's Y" / "More than just X"
- "In today's fast-paced world…" / "In an era where…"
- "Not only… but also…"
- "Whether you're X or Y…"
- "When it comes to…"
- "At its core…" / "At the end of the day…"
- "It's worth noting that…" / "It's important to note…"

## Banned transitions & fillers

Moreover, Furthermore, Additionally, Notably, In conclusion, To summarize,
That said. → Cut them or use a plain "and" / "but" / "so".

Hollow intensifiers: truly, genuinely, really, quite frankly, to be honest.

Hedge stacking: "could potentially", "may eventually", "might ultimately".
Pick one or drop it.

## Structure & formatting tells

- **Em dashes**: max ~1 per 1,000 words. Prefer commas, periods, parentheses.
- **Rule of three**: stop padding ("fast, efficient, and scalable"). Use two or four, or a sentence.
- **Uniform paragraphs**: don't write everything in 3–4 sentence blocks. Include 1–2 sentence ones.
- **Bullet overuse**: convert list-y prose back to prose; bullets only for real lists.
- **Bold overuse**: one bolded phrase per section, max.
- **Title Case Headings** → use sentence case.
- **Inline-header lists**: "**Performance:** Performance improved…" repeats the label. Cut the echo.

## Content & chatbot tells

- Chatbot artifacts: "Great question!", "Certainly!", "I hope this helps!", "Let's dive in!"
- Vague attribution: "Experts believe", "Studies show", "Research suggests". Name the source.
- Generic closers: "The future looks bright", "Only time will tell", "As we move forward".
- Significance inflation: "marks a pivotal moment", "watershed moment" for routine things.
- Synonym cycling: "developers… engineers… builders… practitioners". Repeat the right word.
- Copula avoidance: "serves as / features / boasts" → just use "is / has".
- Superficial -ing analysis: "showcasing a new era", "reflecting decades of investment". State the fact instead.
- Rhetorical-question openers: "What does this mean?" / "Why should you care?". Get to the point.
- Reasoning-chain leaks: "Let me think step by step", "Breaking this down", "Step 1:".

## Quick self-check before shipping prose

- Could I delete this sentence without losing information? Then delete it.
- Did I take a position, or just hedge?
- Are there banned words / em-dash overuse / rule-of-three padding?
- Do my paragraphs vary in length, or are they all the same shape?
- Did I show the point, or just announce that there is one?
