# -*- coding: utf-8 -*-
import os
import core.cio
import core.data
import core.perf
import core.utils
import inference.mv
import inference.ds
import inference.mmli
import inference.mmld
import inference.ocmc
import inference.doc

#in_resp_path = 'D:/Github/datasets/aircrowd6.response.txt'
#in_gold_path = 'D:/Github/datasets/aircrowd6.gold.txt'

#in_resp_path = 'D:/Github/datasets/rte.response.txt'
#in_gold_path = 'D:/Github/datasets/rte.gold.txt'

#in_resp_path = 'D:/Github/datasets/valence5.response.txt'
#in_gold_path = 'D:/Github/datasets/valence5.gold.txt'

#in_resp_path = 'D:/Github/datasets/valence7.response.txt'
#in_gold_path = 'D:/Github/datasets/valence7.gold.txt'

in_resp_path = 'D:/Github/datasets/synth.resp'
in_gold_path = 'D:/Github/datasets/synth.gold'

#in_resp_path = 'D:/zcrom/Output/affective-ml.resp'
#in_gold_path = 'D:/zcrom/Output/affective-ml.gold'

#out_resp_path = 'D:/Github/datasets/aircrowd6.resp'
#out_gold_path = 'D:/Github/datasets/aircrowd6.gold'

dataset = core.cio.load_file(in_resp_path, in_gold_path)
map_path  = os.path.splitext(in_resp_path)[0] + '.map'
core.cio.save_map_file(dataset, map_path)

#core.cio.save_file(dataset, out_resp_path, out_gold_path)

mv = inference.mv.MVModel()
mv.infer(dataset)
eval = core.perf.Evaluation(dataset)
num_label = dataset.get_label_id_size()
for label_id in range(1, num_label + 1):
    print('acc on label (' + str(label_id) +'): '+ str(eval.get_accuracy_on_label(label_id)))
print('MV acc: ' + str(eval.get_accuracy()) + ' subset acc: ' + str(eval.get_subset_accuracy()))

maxround = 20
soft = True

ds = inference.ds.DSModel(maxround)
ds.infer(dataset, soft)
eval = core.perf.Evaluation(dataset)
num_label = dataset.get_label_id_size()
for label_id in range(1, num_label + 1):
    print('DS acc on label (' + str(label_id) +'): '+ str(eval.get_accuracy_on_label(label_id)))
print('DS acc: ' + str(eval.get_accuracy()) + ' subset acc: ' + str(eval.get_subset_accuracy()))

ocmc = inference.ocmc.OCMCModel(maxround)
ocmc.infer(dataset, soft)
eval = core.perf.Evaluation(dataset)
num_label = dataset.get_label_id_size()
for label_id in range(1, num_label + 1):
    print('OCMC acc on label (' + str(label_id) +'): '+ str(eval.get_accuracy_on_label(label_id)))
print('OCMC acc: ' + str(eval.get_accuracy()) + ' subset acc: ' + str(eval.get_subset_accuracy()))


mmli = inference.mmli.MMLIModel(maxround)
mmli.infer(dataset, soft)
eval = core.perf.Evaluation(dataset)
num_label = dataset.get_label_id_size()
for label_id in range(1, num_label + 1):
    print('MMLI acc on label (' + str(label_id) +'): '+ str(eval.get_accuracy_on_label(label_id)))
print('MMLI acc: ' + str(eval.get_accuracy()) + ' subset acc: ' + str(eval.get_subset_accuracy()))

R=4
mmld = inference.mmld.MMLDModel(R, maxround)
omega = [None]
rlist = core.utils.gen_rand_sum_one(R)
print(rlist)
for r in rlist:
    omega.append(r)
mmld.set_omega(omega)
mmld.set_converge_rate(0.01)
mmld.infer(dataset, soft)
eval = core.perf.Evaluation(dataset)
num_label = dataset.get_label_id_size()
for label_id in range(1, num_label + 1):
    print('MMLD acc on label (' + str(label_id) +'): '+ str(eval.get_accuracy_on_label(label_id)))
print('MMLD acc: ' + str(eval.get_accuracy()) + ' subset acc: ' + str(eval.get_subset_accuracy()))

doc = inference.doc.DOCModel(R, maxround)
doc.set_omega(omega)
doc.set_converge_rate(0.01)
doc.infer(dataset, soft)
eval = core.perf.Evaluation(dataset)
num_label = dataset.get_label_id_size()
for label_id in range(1, num_label + 1):
    print('DOC acc on label (' + str(label_id) +'): '+ str(eval.get_accuracy_on_label(label_id)))
print('DOC acc: ' + str(eval.get_accuracy()) + ' subset acc: ' + str(eval.get_subset_accuracy()))
