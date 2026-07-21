import json, numpy as np
 
def compute_prob_field(agent_id):
    with open(f'corpus/{agent_id}_traces.json', encoding='utf-8') as f:
        traces = json.load(f)
    
    # Satisfaction rate = fraction of traces where postcondition held
    p = sum(t['post_satisfied'] for t in traces) / len(traces)
    
    # Latency (if you recorded it -- add time.time() calls to collect_traces.py)
    # If you did not record latency, use placeholder 0 for now
    latencies = [t.get('latency_s', 0) for t in traces if t.get('latency_s', 0) > 0]
    if latencies:
        log_lat = np.log(latencies)
        mu, sigma = float(np.mean(log_lat)), float(np.std(log_lat))
        p95 = float(np.percentile(latencies, 95)) * 1000  # convert to ms
    else:
        mu, sigma, p95 = 0.0, 0.0, 0.0
    
    print(f'{agent_id}: p={p:.2f}, mu={mu:.2f}, sigma={sigma:.2f}, p95={p95:.0f}ms')
    return {'satisfaction_rate': round(p, 2), 'confidence_delta': 0.05,
            'min_sample_k': len(traces),
            'latency_lognormal': {'mu': round(mu,2), 'sigma': round(sigma,2)},
            'p95_latency_ms': round(p95)}
 
# Run for all agents
for i in range(1, 21):
    aid = f'A{i:02d}'
    try:
        result = compute_prob_field(aid)
    except FileNotFoundError:
        print(f'{aid}: traces not found yet')

