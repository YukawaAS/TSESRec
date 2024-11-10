import torch


def get_recall(indices, targets):
    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices).nonzero()
    if len(hits) == 0:
        return 0
    n_hits = hits.shape[0]
    recall = float(n_hits) / targets.size(0)
    return recall


def get_mrr(indices, targets):
    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices).nonzero()
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks).data / targets.size(0)
    return mrr


def metrics(logits, targets, padd_idx, k=20):
    logits = logits[:,:-1,:].reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)

    masks = targets != padd_idx
    logits, targets = logits[masks], targets[masks]

    _, indices = torch.topk(logits, k, -1)
    recall = get_recall(indices, targets)
    mrr = get_mrr(indices, targets)
    return {f'hit@{k}':recall, f'mrr@{k}': mrr}
