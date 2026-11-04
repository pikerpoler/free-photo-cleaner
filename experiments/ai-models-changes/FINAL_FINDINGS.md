# Final findings

The app should ship four scorers. All four are trained on the 53-feature vector, with balanced class weights, and all four can take a later batch after the earlier photos are gone.

1. Logistic regression with a stored Hessian.
2. CatBoost, continued from the previous model.
3. Both MLPs: 53→16→1 and 53→128→32→1.
4. Gaussian naive Bayes.

Logistic regression is the one that still matches a full refit when the labels arrive in short, uneven sessions. CatBoost is the strongest score when the sessions are about 200 photos each. The two MLPs rank with those two on sessions of 200, and the extra width does not pull ahead of the small net. Gaussian naive Bayes is unchanged by session size. Its ranking is weaker than the other three, and its probabilities are not usable as a keep/delete cutoff (log loss about 2.7). It is in the app as a simple incremental baseline.

## Data and features

The phone export is `/Users/nadavnissim/Downloads/photocleaner-export`. Of 691 labeled photos, 6 truncated files were skipped, leaving 685 (413 keep, 272 delete). Delete rate 0.397. Labels are keep = 0, delete = 1. These 685 are photos this user already marked, out of about 13,000 in the catalog, so the scores describe that labeled slice.

The vector is the current 53-feature schema. Live Photo, portrait, panorama, HDR, burst, and the GPS flag are dropped. WhatsApp stays, even though it is constant on this export. The detector is Ultralytics YOLO11n standing in for YOLOX-Nano, confidence 0.25, 11 category buckets. Faces are OpenCV YuNet standing in for Apple Vision, score threshold 0.5. Text was not run, so the two text slots are 0. Screenshot, WhatsApp, aspect ratio, and the other metadata bits are filled from the file or the catalog when they exist.

Cached arrays live in `experiments/ai-models-changes/output/`. `incremental_study.py` is the first two-session run. `fractured_study.py` is the run that compares two sessions of 200 with the uneven schedule, on the same photos.

## How a session is trained

Each seed draws a stratified test set of 200 and a stratified training pool of 400 from the 685. The same test photos are scored after every update. The same 400 training photos are used for every schedule in that seed. Eighty-five photos are unused. Ten seeds. The draws are random slices of this labeled library, at the same delete rate (about 0.40 in every slice). They are not a later calendar day.

Three schedules:

- **Two sessions.** Train on 200. Those rows are then gone. Update from the saved model and the next 200 only.
- **Uneven sessions.** The same 400, in this order: 20, 20, 50, 30, 70, 50, 160. Each batch is gone before the next one arrives.
- **Replay.** One fit on all 400. This is the score if the feature rows were kept after the photo files were deleted.

Within a batch, class weights are balanced on that batch: a class with count `n_c` gets weight `n / (2 n_c)`.

Logistic regression and both MLPs standardize with a guarded z-score: if a column's training standard deviation is 0, the scale is 1, so the column centers to 0 instead of becoming NaN. WhatsApp and the two text slots are constant here, so this guard matters. The scaler is fit on the first batch and then frozen. A new scale would change what the saved weights mean. Trees, including CatBoost, use the raw vector. Gaussian naive Bayes uses the same frozen guarded scaler.

Boosters, including CatBoost, may add anywhere from 0 to 40 new trees per batch. The count is chosen on an inner 20% of that batch, by log loss. The chosen count is then fit again on the whole batch. Zero is allowed, so a batch that does not help the inner slice leaves the saved model alone. The first fit uses the same rule to choose its own tree count, up to 40.

The MLPs are Adam, learning rate 0.001, weight decay 0.001, batch size 32, patience 8, at most 80 epochs. The saved weights are the epoch with the best validation log loss on an inner 20% of the current batch. When a model already exists, epoch 0 is those incoming weights, and it wins if another pass does not improve that inner slice. The inner 20% is not trained on. A check on seed 0 at learning rates 0.0003 and 0.0001 did not help the wide net, so the reported run stays at 0.001. Across the ten seeds the small net keeps about 27 then 34 epochs on the two sessions of 200, and the uneven batches average between 12 and 39 epochs. The wide net keeps about 8 or 9 epochs on a session of 200, and 11 to 23 on the uneven batches. Some individual batches keep epoch 0.

Logistic regression's first batch is L2 logistic regression, C = 1, L-BFGS, fit on every row of the batch. It stores the weights, the intercept, and the 54×54 Hessian of that objective (the intercept is included, and the L2 penalty is included on the weights only). Each later batch minimizes its own weighted log loss plus that quadratic, then adds the new batch's curvature into the Hessian. The L2 identity is added once, on the first batch. What has to be saved is the weights, the scaler, and that matrix.

CatBoost uses depth 3, learning rate 0.1, and `init_model` to continue. Early stopping on the inner slice picks the number of new trees, and that count is refit on the whole batch.

Gaussian naive Bayes calls `partial_fit` once per batch. The class counts, means, and variances are the whole memory. Order does not change the fit, apart from the small difference in per-batch class weights.

## What the numbers mean

AUC ranks by P(delete). It is the number for a swipe queue. Delete F1, precision, and recall are the delete class at a cutoff of 0.5. Log loss is the probability score. The ± on AUC is the spread across the ten seeds. Paired gaps are the same test photos, one schedule minus another, and those are the comparisons to trust.

On an earlier single-training study, with an 80/20 split of all 685 (about 550 training labels) and this same 53-feature schema, balanced logistic regression was AUC 0.82 and delete F1 0.70. Balanced trees and both MLPs of that study (53→16→1 and 53→64→32→1) sat in the same place, AUC about 0.82–0.83. The session numbers below, around 0.81 after 400 labels, are the same neighborhood.

## Two sessions of 200

First run, before CatBoost and the 128-wide net. Incremental means the second session updated the saved model. Session 2 only throws that model away and trains on the latest 200. Replay refits on all 400.

| model | data | AUC | delete F1 | delete recall | log loss |
| --- | --- | ---: | ---: | ---: | ---: |
| logistic, Hessian | session 1, 200 | 0.791 ± 0.023 | 0.647 | 0.670 | 0.622 |
| logistic, Hessian | incremental | 0.807 ± 0.021 | 0.668 | 0.709 | 0.552 |
| logistic, Hessian | session 2 only | 0.778 ± 0.022 | 0.651 | 0.677 | 0.681 |
| logistic, Hessian | replay, 400 | 0.810 ± 0.021 | 0.665 | 0.708 | 0.555 |
| MLP 53→16→1 | session 1, 200 | 0.805 ± 0.027 | 0.653 | 0.676 | 0.541 |
| MLP 53→16→1 | incremental | 0.809 ± 0.023 | 0.662 | 0.719 | 0.555 |
| MLP 53→16→1 | session 2 only | 0.785 ± 0.026 | 0.651 | 0.694 | 0.610 |
| MLP 53→16→1 | replay, 400 | 0.811 ± 0.021 | 0.664 | 0.713 | 0.532 |
| MLP 53→64→32→1 | session 1, 200 | 0.803 ± 0.024 | 0.651 | 0.670 | 0.560 |
| MLP 53→64→32→1 | incremental | 0.812 ± 0.023 | 0.672 | 0.730 | 0.547 |
| MLP 53→64→32→1 | replay, 400 | 0.816 ± 0.022 | 0.654 | 0.689 | 0.529 |
| sklearn boosted trees | session 1, 200 | 0.791 ± 0.031 | 0.633 | 0.662 | 0.542 |
| sklearn boosted trees | incremental | 0.802 ± 0.024 | 0.653 | 0.672 | 0.538 |
| sklearn boosted trees | session 2 only | 0.804 ± 0.030 | 0.663 | 0.680 | 0.528 |
| sklearn boosted trees | replay, 400 | 0.812 ± 0.021 | 0.678 | 0.747 | 0.522 |

The Hessian update matches the refit. Paired incremental-minus-replay AUC is −0.002 ± 0.002. Log loss improves on all 10 seeds against session 1 (−0.069). Against a model trained only on the second session, incremental AUC is higher on all 10 seeds (+0.030 ± 0.020), and precision in the top 20% of the queue stays at 0.82 where the session-2 model falls to 0.77. The weight vector moves (L2 step 2.13) and lands on the refit.

The small MLP's incremental AUC is 0.809 against a replay of 0.811 (paired −0.002 ± 0.013). Discarding the first network and training on session 2 alone drops AUC to 0.785. The continuation runs about 33 epochs and keeps epoch 25. Log loss is 0.555 against 0.532 for the replay, so the ranking holds and the probability scale is a little off. The 64-wide net in this first run has the same shape: paired AUC +0.009 ± 0.006 over session 1, and −0.004 ± 0.013 against the replay. It stops around epoch 6 of the continuation. The app's wide net is the 128-wide one from the next run, not this 64-wide net. The 64-wide result is here because it was the width in the first session study. It did not beat the small net by a meaningful margin.

Sklearn boosted trees can append trees, and on this library that update does not collect the first session. Incremental AUC 0.802 against 0.804 for a model trained only on session 2. The replay reaches delete recall 0.75, against 0.67 for the appended trees. The count of new trees, chosen on 40 validation photos, came out 0, 1, or about 40. A single decision tree cannot be updated at all. A second `fit` replaces it.

## Uneven sessions, same 400 photos

This run adds CatBoost, Gaussian naive Bayes, and the 128-wide MLP, and it repeats logistic regression, the small MLP, and the sklearn trees so the schedules share a split. AUC after all 400 labels:

| model | two sessions | uneven | replay | two-session F1 | uneven F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| logistic, Hessian | 0.808 ± 0.021 | 0.812 ± 0.015 | 0.810 ± 0.021 | 0.664 | 0.664 |
| CatBoost, continued | 0.820 ± 0.019 | 0.791 ± 0.025 | 0.818 ± 0.022 | 0.681 | 0.657 |
| MLP 53→16→1 | 0.814 ± 0.025 | 0.802 ± 0.027 | 0.811 ± 0.022 | 0.662 | 0.654 |
| MLP 53→128→32→1 | 0.811 ± 0.023 | 0.797 ± 0.027 | 0.810 ± 0.017 | 0.657 | 0.645 |
| sklearn boosted trees | 0.804 ± 0.021 | 0.765 ± 0.026 | 0.813 ± 0.018 | 0.656 | 0.639 |
| Gaussian NB | 0.782 ± 0.030 | 0.782 ± 0.030 | 0.782 ± 0.029 | 0.589 | 0.589 |
| LightGBM, continued | 0.814 ± 0.014 | 0.773 ± 0.024 | 0.815 ± 0.022 | 0.676 | 0.636 |
| adaptive random forest | 0.796 ± 0.027 | 0.775 ± 0.029 | 0.788 ± 0.017 | 0.596 | 0.604 |
| SGD logistic | 0.790 ± 0.022 | 0.726 ± 0.043 | 0.790 ± 0.029 | 0.661 | 0.602 |
| Hoeffding tree | 0.774 ± 0.035 | 0.782 ± 0.030 | 0.774 ± 0.034 | 0.589 | 0.590 |

Logistic regression is the same model on all three schedules. Paired uneven-minus-two-sessions AUC is +0.004 ± 0.009. Paired against replay it is +0.002 ± 0.010. Test AUC along the uneven path is 0.634, 0.679, 0.729, 0.753, 0.767, 0.778, 0.812 after 20, 40, 90, 120, 190, 240, and 400 labels.

CatBoost is the best two-session score in this round: AUC 0.820, delete F1 0.681, delete recall 0.742, log loss 0.519, against a replay of 0.818. The seven-batch schedule costs that advantage on every seed (paired −0.029 ± 0.016). The path dips when the second batch of 20 arrives (0.639 then 0.629) and ends at 0.791. LightGBM, which is not in the app list, does the same thing one step lower: 0.814 on two sessions, 0.773 uneven, lower on all 10 seeds. Sklearn trees fall from 0.804 to 0.765, and the path goes from 0.647 after the first 20 to 0.617 after 40.

The 128-wide net matches the small one. Two sessions are 0.811 against 0.814. On the uneven schedule its log loss is 0.625, against 0.573 for the small net and 0.543 for its own replay. Both stay inside the seed noise of their two-session scores (small −0.011 ± 0.024, wide −0.015 ± 0.036).

Gaussian naive Bayes is the same fit on every schedule. AUC 0.782, delete F1 0.589, and it calls delete on about 26% of the test photos. Log loss is 2.7. An earlier full-data run had log loss 2.9. The ranking is real and weaker than logistic regression. The probabilities are not a cutoff.

## Models that were tried and are not in the app list

SGD logistic (`SGDClassifier`, log loss, adaptive step size, `eta0` 0.05, `partial_fit`) matches its own refit on two sessions of 200, at AUC 0.790 and log loss 0.67. The uneven schedule breaks it: AUC 0.726, log loss 1.37. The Hessian logistic is the linear model that survives that schedule. River's online logistic regression is the same kind of update, so it was not run separately.

A Hoeffding tree with grace period 20 and delta 0.01 never grew. A pass over 400 rows of seed 0 left a single leaf, so the score is the naive Bayes in that leaf. The library defaults (grace period 200, delta 1e-7) are for streams of millions. Forcing splits on that same pass grew a 5-node tree and the AUC fell from 0.747 to about 0.70. An adaptive random forest of ten trees does rank (two-session AUC 0.796, log loss about 0.53) and is the best of the set at 20 labels (AUC 0.684). By 400 labels it is behind logistic regression and CatBoost, and delete F1 stays near 0.60 because it calls delete on about 30% of photos.

LightGBM continued from the previous model matches CatBoost's pattern and scores a bit lower. XGBoost would be the same kind of continued boosting.

## What each app model has to keep

- **Logistic, Hessian.** Coefficient vector, intercept, guarded scaler (mean and scale from the first batch), and the 54×54 Hessian. Later batches do not need the old rows.
- **CatBoost.** The model itself, continued with `init_model`. Up to 40 new trees per session, the count chosen on an inner 20% of that session. On sessions of about 200 this matches a refit. On the short uneven schedule it gives back about 0.03 AUC, so a user who labels in tiny bursts is better served by the logistic model, or by keeping the feature rows and refitting CatBoost.
- **Both MLPs.** The weights at the best validation epoch, and the guarded scaler from the first batch. Later sessions keep training at learning rate 0.001 and may stop at epoch 0. The small net is 53→16→1. The wide net is 53→128→32→1.
- **Gaussian naive Bayes.** Class counts, per-class means and variances, and the guarded scaler. One `partial_fit` per session. Use it as a ranking baseline. Do not threshold its probabilities at 0.5 and expect a calibrated delete rate. On this library it calls delete on about 26% of photos and the log loss is about 2.7.
