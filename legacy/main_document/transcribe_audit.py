#!/usr/bin/env python3
"""转录用户人工 audit 结果（71 条——从逐条表）"""
import json
 
# (game, turn, human_intervention, student_acceptable, teacher_vs_student)
rows = [
(1215,29,'Skip','Yes','Equivalent'),(1490,17,'Skip','Yes','Equivalent'),(1380,1,'Skip','Yes','Equivalent'),
(3518,3,'Skip','Yes','Worse'),(3327,44,'Strong','No','Better'),(2309,18,'Weak','Yes','Equivalent'),
(542,31,'Strong','No','Better'),(424,12,'Skip','Yes','Equivalent'),(2076,21,'Weak','Yes','Equivalent'),
(1380,3,'Skip','Yes','Worse'),(2078,2,'Weak','Yes','Better'),(674,28,'Skip','Yes','Equivalent'),
(2568,41,'Weak','Yes','Equivalent'),(3228,44,'Weak','Yes','Equivalent'),(674,39,'Weak','Yes','Better'),
(966,1,'Skip','Yes','Equivalent'),(3015,3,'Skip','Yes','Worse'),(2245,4,'Skip','Yes','Equivalent'),
(904,3,'Skip','Yes','Equivalent'),(1914,20,'Strong','No','Better'),(871,1,'Skip','Yes','Equivalent'),
(2033,8,'Skip','Yes','Equivalent'),(396,1,'Skip','Yes','Equivalent'),(2156,1,'Skip','Yes','Equivalent'),
(2566,1,'Skip','Yes','Equivalent'),(3228,38,'Skip','Yes','Equivalent'),(612,3,'Skip','Yes','Worse'),
(3467,46,'Strong','No','Better'),(984,15,'Strong','No','Better'),(542,27,'Weak','Yes','Equivalent'),
(2309,3,'Skip','Yes','Worse'),(3088,23,'Weak','Yes','Equivalent'),(2827,41,'Skip','Yes','Equivalent'),
(647,24,'Strong','No','Equivalent'),(2309,14,'Skip','Yes','Equivalent'),(2566,2,'Skip','Yes','Equivalent'),
(2566,3,'Skip','Yes','Equivalent'),(670,1,'Skip','Yes','Equivalent'),(1340,5,'Skip','Yes','Equivalent'),
(67,3,'Skip','Yes','Equivalent'),(1490,13,'Skip','Yes','Equivalent'),(2827,36,'Skip','Yes','Equivalent'),
(1171,43,'Weak','Yes','Equivalent'),(647,10,'Skip','Yes','Equivalent'),(3327,49,'Strong','No','Better'),
(984,43,'Skip','Yes','Equivalent'),(612,12,'Skip','Yes','Equivalent'),(3327,45,'Strong','No','Better'),
(3327,14,'Strong','No','Better'),(338,3,'Skip','Yes','N/A'),(2156,6,'Skip','Yes','N/A'),
(3472,5,'Skip','Yes','Worse'),(612,14,'Skip','Yes','Equivalent'),(542,38,'Strong','No','Better'),
(2204,7,'Strong','No','Better'),(3327,30,'Strong','No','Better'),(1521,3,'Weak','Yes','Equivalent'),
(3327,22,'Strong','No','Better'),(542,43,'Strong','No','Better'),(647,22,'Skip','Yes','Worse'),
(1171,4,'Strong','No','Better'),(2204,19,'Strong','No','Better'),(647,21,'Skip','Yes','Equivalent'),
(1957,21,'Strong','No','Better'),(2204,34,'Strong','No','Better'),(2204,20,'Strong','No','Better'),
(542,26,'Strong','No','Better'),(1198,38,'Strong','No','Better'),(2204,5,'Strong','No','Better'),
(2204,45,'Strong','No','Better'),(2204,14,'Strong','No','Better'),
]
out = [{'game_turn': f'{g}/{t}', 'human_intervention': i, 'human_acceptable': a, 'human_teacher_vs': tv}
       for g, t, i, a, tv in rows]
with open('/root/data/alfworld/opd/a1/a3/run1/sage_audit_human_results.jsonl', 'w') as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'transcribed: {len(rows)} rows')
from collections import Counter
print('intervention:', dict(Counter(r[2] for r in rows)))
print('acceptable:', dict(Counter(r[3] for r in rows)))
print('teacher_vs:', dict(Counter(r[4] for r in rows)))
