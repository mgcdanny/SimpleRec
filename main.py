import pandas as pd


def llr(k):
    '''
        Compute loglikelihood ratio see
        http://tdunning.blogspot.de/2008/03/surprise-and-coincidence.html
        And
        https://github.com/apache/mahout/blob/4f2108c576daaa3198671568eaa619266e787b1a/math/src/main/java/org/apache/mahout/math/stats/LogLikelihood.java#L100
        And https://en.wikipedia.org/wiki/G-test
    '''

    def H(k):
        N = k.values.sum()
        wtf = pd.np.log(k / N + (k == 0).astype(int))
        return (k / N * wtf).values.sum()

    return 2 * k.values.sum() * (H(k) - H(k.sum(0)) - H(k.sum(1)))


def compute_scores(A, B, skip_diagonal=False):
    '''
        Compute the scores for a primary and secondary action (across all items)
        'A' is the user x item matrix of the primary action
        'B' is the user x item matrix of a secondary action
        the result is a dataframe where
        'primary_item' is the item associated with the primary event (ie 'buy')
        'secondary_item' is the item associated with the secondary event (ie 'click')
        'score' is the log likelihood score representing the strength of
        association (the higher the score the stronger association)
        For example, people who 'primary action' item_A do 'secondary action'
        item_B with strength 'score'
        Loosely based on:
            https://github.com/apache/mahout/blob/4f2108c576daaa3198671568eaa619266e787b1a/math-scala/src/main/scala/org/apache/mahout/math/cf/SimilarityAnalysis.scala#L312
    '''

    # We ignore the original interaction value and create a binary (binarize) 0-1 matrix
    # as we only consider whether interactions happened or did not happen
    # only consider action B for users where action A occured

    A = (A != 0).astype(int)
    B = (B != 0).astype(int)
    AtB = A.loc[B.index, B.columns].transpose().dot(B)
    numInteractionsWithAandB = AtB
    numInteractionsWithA = A.sum()
    numInteractionsWithB = B.sum()
    # Total number of interactions is
    # total number of users where primary event occurs
    numInteractions = len(A)

    K11 = numInteractionsWithAandB
    K12 = numInteractionsWithAandB.rsub(numInteractionsWithA, axis=0).dropna()
    K21 = numInteractionsWithAandB.rsub(numInteractionsWithB, axis=1)
    K22 = numInteractions + numInteractionsWithAandB.sub(
        numInteractionsWithB, axis=1).sub(
            numInteractionsWithA, axis=0)

    the_data = zip(
        K11.apply(lambda x: x.index + '_' + x.name).values.flatten(),
        K11.values.flatten(), K12.values.flatten(), K21.values.flatten(),
        K22.values.flatten())

    container = []
    for name, k11, k12, k21, k22 in the_data:
        item_A, item_B = name.split('_')
        if k11 != 0 and not (skip_diagonal and item_A == item_B):
            df = pd.DataFrame([[k11, k12], [k21, k22]])
            score = llr(df)
        else:
            score = 0  # Warning! while llr score could be calculated, for cooccurance purposes, it doesn't makes sense to compute llr when cooccurnace (k11) is zero
        container.append((item_A, item_B, score))
    return pd.DataFrame(
        container, columns=['primary_item', 'secondary_item',
                            'score']).sort_values(
                                ['primary_item', 'score'],
                                ascending=[True, False])


def train(raw_data, primary_action):
    '''
        this is like the 'main' funciton: takes a dataset and returns a dataframe with LLR scores

        raw_data is a dataframe with the columns: user, action, item

        primary_action is the action from raw_data that we want to determine associations for

        'A' is the matrix of primary actions
        'B' is a matrix of secondary actions

    '''

    #  pretty sure we only want to keep users and user metadata for only the users where the primary action occurs
    #  not sure where this happens int he Mahout code though...
    users_who_did_primary_action = pd.unique(
        raw_data.loc[raw_data.action == primary_action, 'user'])
    data = raw_data.loc[raw_data.user.isin(users_who_did_primary_action), :]

    freq = data.groupby(['user', 'action',
                         'item']).size().to_frame('freq').reset_index()
    freq_actions = freq.groupby('action')

    A = freq_actions.get_group(primary_action).pivot(
        index='user', columns='item', values='freq').fillna(0)

    cco_results = []
    for action, matrix in freq_actions:
        skip_diagonal = primary_action == action
        B = matrix.pivot(index='user', columns='item', values='freq').fillna(0)
        scores = compute_scores(A, B, skip_diagonal)
        scores['primary_action'] = primary_action
        scores['secondary_action'] = action
        cco_results.append(scores)
    all_data = pd.concat(cco_results, ignore_index=True)

    return all_data[[
        'primary_action', 'primary_item', 'secondary_action', 'secondary_item',
        'score'
    ]]


if __name__ == '__main__':
    '''
        These unit tests are the same as the Apache Mahout unit tests per:
            https://github.com/apache/mahout/blob/08e02602e947ff945b9bd73ab5f0b45863df3e53/spark/src/test/scala/org/apache/mahout/cf/SimilarityAnalysisSuite.scala#L49
            https://github.com/apache/mahout/blob/08e02602e947ff945b9bd73ab5f0b45863df3e53/math/src/test/java/org/apache/mahout/math/stats/LogLikelihoodTest.java#L50
            https://github.com/apache/mahout/blob/4f2108c576daaa3198671568eaa619266e787b1a/math/src/main/java/org/apache/mahout/math/stats/LogLikelihood.java#L1

    '''

    # test compute_scores
    a = pd.DataFrame(
        [(1, 1, 0, 0, 0), (0, 0, 1, 1, 0), (0, 0, 0, 0, 1), (1, 0, 0, 1, 0)],
        columns=['a', 'b', 'c', 'd', 'e'])

    b = pd.DataFrame(
        [(1, 1, 1, 1, 0), (1, 1, 1, 1, 0), (0, 0, 1, 0, 1), (1, 1, 0, 1, 0)],
        columns=['a', 'b', 'c', 'd', 'e'])

    AtAControl = pd.DataFrame(
        [(0.0, 1.7260924347106847, 0.0, 0.0, 0.0),
         (1.7260924347106847, 0.0, 0.0, 0.0, 0.0),
         (0.0, 0.0, 0.0, 1.7260924347106847, 0.0),
         (0.0, 0.0, 1.7260924347106847, 0.0, 0.0),
         (0.0, 0.0, 0.0, 0.0, 0.0)])\
        .round(5)\
        .as_matrix()

    AtBControl = pd.DataFrame(
        [(1.7260924347106847, 1.7260924347106847, 1.7260924347106847, 1.7260924347106847, 0.0),
         (0.6795961471815897, 0.6795961471815897, 0.6795961471815897, 0.6795961471815897, 0.0),
         (0.6795961471815897, 0.6795961471815897, 0.6795961471815897, 0.6795961471815897, 0.0),
         (1.7260924347106847, 1.7260924347106847, 1.7260924347106847, 1.7260924347106847, 0.0),
         (0.0, 0.0, 0.6795961471815897, 0.0, 4.498681156950466)])\
        .round(5)\
        .as_matrix()

    ata = compute_scores(a, a, True).pivot(
        index='primary_item', columns='secondary_item',
        values='score').round(5).as_matrix()
    atb = compute_scores(a, b, False).pivot(
        index='primary_item', columns='secondary_item',
        values='score').round(5).as_matrix()

    assert pd.np.array_equal(ata, AtAControl)
    assert pd.np.array_equal(atb, AtBControl)

    # test llr

    assert 2.773 == round(llr(pd.DataFrame([[1, 0], [0, 1]])), 3)
    assert 27.726 == round(llr(pd.DataFrame([[10, 0], [0, 10]])), 3)
    assert 39.331 == round(llr(pd.DataFrame([[5, 1995], [0, 100000]])), 3)
    assert 4730.737 == round(
        llr(pd.DataFrame([[1000, 1995], [1000, 100000]])), 3)
    assert 5734.343 == round(
        llr(pd.DataFrame([[1000, 1000], [1000, 100000]])), 3)
    assert 5714.932 == round(
        llr(pd.DataFrame([[1000, 1000], [1000, 99000]])), 3)
