# -*- coding: utf-8 -*-
import os
import core.cio
import core.data
import core.perf
import inference.mv
import inference.ds
import inference.mmli
import inference.mmld

#in_resp_path = 'D:/Github/datasets/aircrowd6.response.txt'
#in_gold_path = 'D:/Github/datasets/aircrowd6.gold.txt'

in_resp_path = 'D:/Github/datasets/synth.resp'
in_gold_path = 'D:/Github/datasets/synth.gold'

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
print('total acc: ' + str(eval.get_accuracy()))

maxround = 20
#ds = inference.ds.DSModel(maxround)
#ds.infer(dataset)
#eval = core.perf.Evaluation(dataset)
#num_label = dataset.get_label_id_size()
#for label_id in range(1, num_label + 1):
#    print('DS acc on label (' + str(label_id) +'): '+ str(eval.get_accuracy_on_label(label_id)))
#print('DS total acc: ' + str(eval.get_accuracy()))

#mmli = inference.mmli.MMLIModel(maxround)
#mmli.infer(dataset)
#eval = core.perf.Evaluation(dataset)
#num_label = dataset.get_label_id_size()
#for label_id in range(1, num_label + 1):
#    print('MMLI acc on label (' + str(label_id) +'): '+ str(eval.get_accuracy_on_label(label_id)))
#print('MMLI total acc: ' + str(eval.get_accuracy()))

mmld = inference.mmld.MMLDModel(4, maxround)
omega = [None]
omega.append(0.15)
omega.append(0.28)
omega.append(0.24)
omega.append(0.33)
mmld.set_omega(omega)
mmld.infer(dataset)
eval = core.perf.Evaluation(dataset)
num_label = dataset.get_label_id_size()
for label_id in range(1, num_label + 1):
    print('MMLD acc on label (' + str(label_id) +'): '+ str(eval.get_accuracy_on_label(label_id)))
print('MMLD total acc: ' + str(eval.get_accuracy()))
