# recommendation_system/diversity.py

import math

def diversity_rerank(candidate_list, question_db, top_n=10, alpha=0.3):
    """
    对 candidate_list(已排序) 做多样性重排, 示例: greedy插入:
      - alpha=相似度惩罚权重, 越大表示越强调多样性(惩罚相似项目).
      - question_db: {qid: { 'vector':[...], ... }, ...}

    算法示意: 依次从头到尾, 对下一个candidate,
    计算与已选集合平均相似度, cost= rank_score - alpha*(avg_sim),
    选 cost 最大者
    """
    if len(candidate_list)<=1:
        return candidate_list[:top_n]

    selected = []
    not_selected = candidate_list[:]

    def get_vector(qid):
        return question_db.get(qid,{}).get("vector",[])

    while not_selected and len(selected)<top_n:
        best_item=None
        best_cost=-9999999
        for cid in not_selected:
            cscore= _score(cid, selected, question_db, alpha, get_vector)
            if cscore>best_cost:
                best_cost=cscore
                best_item=cid
        if best_item:
            selected.append(best_item)
            not_selected.remove(best_item)
        else:
            break

    return selected

def _score(qid, selected, qdb, alpha, get_vector_func):
    """
    cost = (pos_in_candidate_list or some base score) - alpha*(average similarity to selected)
    这里简化: base_score= length - index  (越前越大),
    or可改为 question_db[qid]["rank_score"]
    """
    # 先做个简化, base_score=1000-index
    # 真实情况需要在多样性重排前保留个score
    # 这里仅演示
    base_score=100.0
    # 计算与selected的平均相似
    if not selected:
        return base_score
    sim_sum=0.0
    for sid in selected:
        sim_sum+=_cosine_similarity(get_vector_func(qid), get_vector_func(sid))
    avg_sim= sim_sum/ len(selected)
    cost = base_score - alpha*avg_sim
    return cost

def _cosine_similarity(vec1, vec2):
    if not vec1 or not vec2:
        return 0.0
    dot=sum(a*b for a,b in zip(vec1,vec2))
    norm1= math.sqrt(sum(a*a for a in vec1))
    norm2= math.sqrt(sum(b*b for b in vec2))
    if not norm1 or not norm2:
        return 0.0
    return dot/(norm1*norm2)
