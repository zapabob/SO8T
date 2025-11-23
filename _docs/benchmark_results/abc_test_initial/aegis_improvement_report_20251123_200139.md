# AEGIS Improvement Analysis Report

**Generated:** 2025-11-23 20:01:39

## Executive Summary

⚠️ **No significant improvement:** -20.0% accuracy change
✅ **Hallucination reduction:** -40.0% hallucination rate

## Detailed Results

### Overall Performance

| Metric | Baseline | Improved | Change |
|--------|----------|----------|--------|
.1%.1%+.1%.1%+.1%.1%+.1%.1%+.1%
### Test Case Analysis

#### Test Case 1
**Question:** Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

**Baseline Model:**
- Response: TIMEOUT
.2f- Correct: ❌
- Reasoning Quality: poor
- Hallucination: ✅

**Improved Model:**
- Response: TIMEOUT
.2f- Correct: ❌
- Reasoning Quality: poor
- Hallucination: ✅

#### Test Case 2
**Question:** A robe takes 2 bolts of blue fiber and half times that much white fiber. How many bolts in total does it take?

**Baseline Model:**
- Response: I would write the equation as:  
$$\text{total fibers} = \boxed{1}\tag{$B_b=2$, $B_w=\frac{1}{2}(2)=1$}$$ 

Here, we are given that one robe takes 2 bolts of blue fiber ($B_b$) and half times the amou...
.2f- Correct: ✅
- Reasoning Quality: good
- Hallucination: ⚠️

**Improved Model:**
- Response: TIMEOUT
.2f- Correct: ❌
- Reasoning Quality: poor
- Hallucination: ✅

#### Test Case 3
**Question:** Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?

**Baseline Model:**
- Response: To calculate Josh's profit from renovating his new property, we need to follow these steps:

1. Calculate the total investment made into the property (the initial purchase price plus repair costs):
  ...
.2f- Correct: ✅
- Reasoning Quality: excellent
- Hallucination: ⚠️

**Improved Model:**
- Response: To calculate Josh's total investment:
Total Investment = Purchase Price + Repair Costs
                     = $80,000 + $50,000
                     = $130,000
To find the increased value of the house...
.2f- Correct: ✅
- Reasoning Quality: fair
- Hallucination: ⚠️

#### Test Case 4
**Question:** Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the evening to follow her daily routine?

**Baseline Model:**
- Response: Let's calculate it step by step:
1. Morning meal = 15 cups
2. Afternoon meal = 25 cups
3. Total feed given before evening = morning + afternoon = 15 cups + 25 cups = 40 cups
4. To follow her daily rou...
.2f- Correct: ❌
- Reasoning Quality: excellent
- Hallucination: ⚠️

**Improved Model:**
- Response: To find out how much feed Wendi needs to give her chickens in the evening, we first add up the amount of feed given during the morning and afternoon: 15 + 25 = 40 cups. Since she feeds them three sepa...
.2f- Correct: ❌
- Reasoning Quality: good
- Hallucination: ⚠️

#### Test Case 5
**Question:** A company produces 420 units of a product per day. The company operates 6 days a week. How many units does the company produce in a week?

**Baseline Model:**
- Response: To calculate this, we need to multiply the number of units produced each day by the number of operating days in a week:

\[ \text{Units per day} = 420 \]
\[ \text{Days per week} = 6 \]
\[ \text{Total ...
.2f- Correct: ✅
- Reasoning Quality: good
- Hallucination: ⚠️

**Improved Model:**
- Response: To find out how many units are produced in one week, we need to multiply the daily production by the number of operating days: 

Units per Week = Units Per Day * Days Operated
= 420 units/day * 6 days...
.2f- Correct: ✅
- Reasoning Quality: fair
- Hallucination: ✅

## Recommendations

### 🔄 Needs Further Investigation
- No significant improvement detected
- Consider different approaches:
  - Reduce Alpha Gate scale factor further (e.g., 0.6)
  - Increase logic training epochs
  - Use different dataset combinations
  - Review SO(8) transformation parameters

## Technical Details

### Configuration Used
- Alpha Gate Scale Factor: 0.8
- Logic Training Epochs: 2
- LoRA Rank: 8
- Training Dataset: GSM8K + Synthetic Logic Data

### Test Methodology
- 9 comprehensive test cases
- Mathematical correctness evaluation
- Reasoning quality assessment
- Hallucination detection
- Response time measurement

