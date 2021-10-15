"""
Bayesian Analysis of a simple button-pressing game, in which 1 button produces a reward 80% of the time,
and another button produces a reward 20& of the time.

We always push button A, and check if we get a reward or not.
Depending on the outcome (reward or not), we update the posteriors for each of our hypotheses.

A hypothesis simply specifies the probability that button A results in a reward.
"""
import random

REWARD_PROB_BTN_A = 0.8
NUM_TRIALS = 10

# the likelihood is the probability that button A produces a reward, given a hypothesis.
# note: this is equivalent to "the probability that button A is the best button, in the long run"
H2LIKELIHOOD = {
    'H2': 0.8,
    'H3': 0.5,
}

# uniform prior
h2prior = {h: 1.0 / float(len(H2LIKELIHOOD)) for h in H2LIKELIHOOD.keys()}

hypotheses = list(H2LIKELIHOOD.keys())

hypothesis2posterior = h2prior.copy()

rewards = []
for trial in range(NUM_TRIALS):

    # we always push button a
    is_reward = random.random() < REWARD_PROB_BTN_A
    rewards.append(is_reward)

    tmp = []
    for h in hypotheses:
        likelihood = H2LIKELIHOOD[h] if is_reward else 1 - H2LIKELIHOOD[h]
        prior = hypothesis2posterior[h]  # previous posterior is now the prior
        p_hd = prior * likelihood  # P(H) * P(D|H)
        tmp.append(p_hd)
    p_hd_sum = sum(tmp)

    # update
    hypothesis2posterior = {h: p_hd / p_hd_sum for h, p_hd in zip(hypotheses, tmp)}

    printout = " ".join([f'P({h})={posterior:.4f}' for h, posterior in zip(hypotheses, hypothesis2posterior.values())])
    print(f'trial={trial:>3} | reward={"T" if is_reward else " "} | {printout}')


actual_reward_prob = sum(rewards) / float(len(rewards))
print(f'actual_reward_prob={actual_reward_prob}')


