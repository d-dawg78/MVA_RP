#### Final Project Submission for the Représentations Parcimonieuses course at the École Normale Supérieure (Master MVA)!

#### Group: Dorian Desblancs, Liam Gonzalez

#### Abstract:

In this project, we tackled the problem of automatic sleep apnea detection proposed by Dreem, a French neuro-technology start-up. We were given a set of signals that are commonly used for manual sleep apnea detection, and a set of masks indicating where and when a apneic event was occurring. The goal of this project: to automatically derive this mask from the set of signals provided, using an algorithm of our choice. Our best solution merged a multi-layer convolutional neural network (CNN) with a Gated Recurrent Unit (GRU) network. We used a Tversky loss function during training. Our network used all eight 1D signals as input, and achieved a final F1-score of 0.5593 on the hold-out set. These results place us near the top of the challenge ranking (currently sixth). This report explores our best method in depth. It also explores some of the other approaches we implemented, and discusses whether these should be further inspected.
