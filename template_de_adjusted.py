v1 = {
    "cls": {
        "sst-5": """Follow the steps to create a better prompt:

1. Compare Prompt 1 and Prompt 2 to find differences.
Prompt 1: "<prompt1>"
Prompt 2: "<prompt2>"

2. Change the different parts:
Example: "comment" → "review" and "sentence" → "phrase".

3. Combine the changed parts with Prompt 3:
Prompt 3: "<prompt3>"

4. Mix the new prompt with the basic prompt:
Basic Prompt: "<prompt0>"

5. Put a final prompt between the <prompt> and </prompt> brackets!
Final Prompt: <prompt> Analyze the review and classify it as terrible, bad, okay, good, or great. </prompt>
"""
    }
}

v2 = {
    "cls": {
        "sst-5": """Create a better prompt for classifying COVID-19 related tweets by following these steps:

1. Find differences between Prompt 1 and Prompt 2.
Prompt 1: "<prompt1>"
Prompt 2: "<prompt2>"

2. Change the parts that are different:
Example: "comment" → "review text".

3. Combine the changes with Prompt 3:
Prompt 3: "<prompt3>"

4. Mix the new prompt with the basic prompt:
Basic Prompt: "<prompt0>"

5. Put a final prompt between the <prompt> and </prompt> brackets!
Final Prompt: <prompt> Evaluate the review text and classify it into terrible, bad, okay, good, or great. </prompt>
"""
    }
}

v3 = {
    "cls": {
        "sst-5": """Follow the steps to create a new prompt:

1. Randomly change parts of Prompt 1 and Prompt 2.
Prompt 1: <prompt1>
Prompt 2: <prompt2>

2. Combine the changed parts with Prompt 3:
Prompt 3: <prompt3>

3. Mix the new prompt with the basic prompt:
Basic Prompt: <prompt0>

4. Put a final prompt between the <prompt> and </prompt> brackets!
Final Prompt: <prompt> Understand the meaning of the review and classify it into terrible, bad, okay, good, or great. </prompt>
"""
    }
}