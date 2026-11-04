# Findings

Ran on the phone export: 691 labeled photos, 6 truncated files skipped, 685 scored (413 keep, 272 delete). Ten stratified 80/20 splits. Detector is Ultralytics YOLO11n, a COCO nano model standing in for YOLOX-Nano. Faces are OpenCV YuNet, standing in for Apple Vision. Text, Live Photo, portrait, panorama, HDR, and burst are 0 in this export.

The honest numbers below pick model size on an inner slice of the training fold. The "plan" numbers pick it on the test fold, which is what the app's chart would show. That first run still includes the GPS flag. The follow-up uses the current 53-feature schema, with GPS and the five subtype flags removed and WhatsApp kept.

## Two sessions

The user labels 200 photos, the model trains, and those rows are gone. A later session labels 200 more. The update is allowed the saved model and the new rows. The same 200 held-out photos are scored after the first session and after the update. Ten seeds. Each slice is a stratified draw from the 685 labeled photos. The second session is another swipe through the same library, at the same delete rate. Eighty-five photos are unused. Delete rate in each slice is 0.40. Balanced class weights. Logistic regression and the MLPs use the guarded scaler from the previous study.

A single decision tree cannot take the second batch. `DecisionTreeClassifier` has no update, and a second `fit` replaces the tree. The boosted trees can: `warm_start` leaves the first session's trees in place and appends trees fit only on the new rows. The number of new trees, anywhere from 0 to 40, is chosen on an inner 20% of the second session. Logistic regression has no `partial_fit`. The first fit stores the weights and the 54×54 Hessian of the objective it minimized, intercept included, L2 included. The second session minimizes its own loss plus that quadratic. The MLP keeps the best validation checkpoint and the session-1 scaler and keeps training. Epoch 0 remains available, so a second pass that does not help the new session's validation loss is dropped. In these 10 seeds that never happened.

Replay trains from scratch on all 400 rows. That is the score if the feature vectors were kept after the files were deleted. The session-2 column trains from scratch on the second 200 and throws the first model away. The ± on AUC is the spread across seeds. The paired lines later are the same test photos, incremental minus the other condition, and those are the comparisons to trust.

| model | data | AUC | delete F1 | delete recall | log loss |
| --- | --- | ---: | ---: | ---: | ---: |
| logistic | session 1, 200 | 0.791 ± 0.023 | 0.647 | 0.670 | 0.622 |
| logistic | incremental | 0.807 ± 0.021 | 0.668 | 0.709 | 0.552 |
| logistic | session 2 only | 0.778 ± 0.022 | 0.651 | 0.677 | 0.681 |
| logistic | replay, 400 | 0.810 ± 0.021 | 0.665 | 0.708 | 0.555 |
| boosted trees | session 1, 200 | 0.791 ± 0.031 | 0.633 | 0.662 | 0.542 |
| boosted trees | incremental | 0.802 ± 0.024 | 0.653 | 0.672 | 0.538 |
| boosted trees | session 2 only | 0.804 ± 0.030 | 0.663 | 0.680 | 0.528 |
| boosted trees | replay, 400 | 0.812 ± 0.021 | 0.678 | 0.747 | 0.522 |
| MLP 53→16→1 | session 1, 200 | 0.805 ± 0.027 | 0.653 | 0.676 | 0.541 |
| MLP 53→16→1 | incremental | 0.809 ± 0.023 | 0.662 | 0.719 | 0.555 |
| MLP 53→16→1 | session 2 only | 0.785 ± 0.026 | 0.651 | 0.694 | 0.610 |
| MLP 53→16→1 | replay, 400 | 0.811 ± 0.021 | 0.664 | 0.713 | 0.532 |
| MLP 53→64→32→1 | session 1, 200 | 0.803 ± 0.024 | 0.651 | 0.670 | 0.560 |
| MLP 53→64→32→1 | incremental | 0.812 ± 0.023 | 0.672 | 0.730 | 0.547 |
| MLP 53→64→32→1 | session 2 only | 0.786 ± 0.028 | 0.656 | 0.705 | 0.596 |
| MLP 53→64→32→1 | replay, 400 | 0.816 ± 0.022 | 0.654 | 0.689 | 0.529 |

Logistic regression accumulates. The incremental scores match the replay: paired AUC −0.002 ± 0.002. Log loss improves on all 10 seeds against session 1 (−0.069) and matches the replay. Against a model trained only on the second session, incremental AUC is higher on all 10 seeds (+0.030 ± 0.020), and precision in the top 20% of the queue stays at 0.82 where the session-2 model falls to 0.77. The weight vector does move (L2 step 2.13, mean absolute change in P(delete) 0.12) and lands on the refit. The gradient of the session-1 objective at the saved weights is 0.015, which is the solver's own stopping tolerance, so the Hessian being stored is the curvature of the fit that was kept. What has to be saved is the weights, the scaler, and that 54×54 matrix.

The small MLP's incremental AUC is 0.809 against a replay of 0.811 (paired −0.002 ± 0.013). The gain over session 1 is +0.004 ± 0.005, higher on 8 of 10 seeds. Training the second session from scratch, with the first network discarded, drops AUC to 0.785 (paired +0.023 ± 0.018 for keeping the checkpoint). The continuation runs about 33 epochs and keeps epoch 25. Log loss is 0.555 against 0.532 for the replay, so the ranking survives and the probability scale is a little worse than refitting. The big MLP is the same pattern with a cleaner gain over session 1: paired AUC +0.009 ± 0.006, higher on all 10 seeds, and −0.004 ± 0.013 against the replay. It stops around epoch 6 of the continuation. Its delete F1 of 0.672 against the replay's 0.654 sits inside a paired spread of ±0.05.

Boosted trees can take the second batch, and on this library the update does not collect the first session. Incremental AUC 0.802 against 0.804 for a model trained only on session 2 (paired −0.002 ± 0.014). The replay on 400 rows reaches delete recall 0.75, against 0.67 for the appended trees (paired −0.075). The old trees are still in the model. Adding zero new trees leaves the scores exactly where session 1 put them, and that happened on one seed. The count of new trees was chosen on 40 validation photos and came out 0, 1, or about 40: 26, 1, 23, 40, 39, 0, 1, 36, 1, 40. Mean 21 new trees, on top of 24 from session 1. A second stage of that size is fit only to the new rows, and 40 validation photos do not decide it reliably. Keeping the 53-vectors and refitting is the tree update that uses both days.

For a user who deletes the files and does not keep the vectors, logistic regression with the Hessian stored beside the weights is the update that matches having kept them. The MLP can continue from its checkpoint and stays with a refit on AUC. The trees should be refit from stored rows. These session-1 AUCs, about 0.79–0.80, and the replay AUCs, about 0.81, sit next to the 0.82 from training on roughly 550 labels in the table below. That is the same shape as the earlier learning curve, which was near the full-data scores by about 300 labels.

## Uneven sessions

Same 400 training photos and the same 200 held-out photos, ten seeds. `two_session` is 200 then 200. `fractured` is 20, 20, 50, 30, 70, 50, 160. `replay` fits those 400 in one go. The wide net here is 53→128→32→1. River's online logistic regression is the same update as `SGDClassifier`, so only the sklearn one is in the table. XGBoost's continued boosting is the same idea as LightGBM's, so only LightGBM is in the table.

The learning rate on both MLPs stayed 0.001. Across the ten seeds the small net keeps about 27 then 34 epochs on the two sessions of 200, and the fractured batches average between 12 and 39 epochs. The 128-wide net keeps about 8 or 9 epochs on a session of 200, and 11 to 23 on the fractured batches. Individual batches do keep epoch 0, which means that batch's inner slice never beat the model it arrived with. On seed 0, dropping the wide net to 0.0003 or 0.0001 left the last batch at one kept epoch and lowered that seed's final AUC from 0.760 to 0.748 and 0.756. A smaller rate was not the fix.

| model | two sessions | fractured | replay | two-session F1 | fractured F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| logistic, Hessian | 0.808 ± 0.021 | 0.812 ± 0.015 | 0.810 ± 0.021 | 0.664 | 0.664 |
| CatBoost, continued | 0.820 ± 0.019 | 0.791 ± 0.025 | 0.818 ± 0.022 | 0.681 | 0.657 |
| LightGBM, continued | 0.814 ± 0.014 | 0.773 ± 0.024 | 0.815 ± 0.022 | 0.676 | 0.636 |
| MLP 53→16→1 | 0.814 ± 0.025 | 0.802 ± 0.027 | 0.811 ± 0.022 | 0.662 | 0.654 |
| MLP 53→128→32→1 | 0.811 ± 0.023 | 0.797 ± 0.027 | 0.810 ± 0.017 | 0.657 | 0.645 |
| sklearn boosted trees | 0.804 ± 0.021 | 0.765 ± 0.026 | 0.813 ± 0.018 | 0.656 | 0.639 |
| adaptive random forest | 0.796 ± 0.027 | 0.775 ± 0.029 | 0.788 ± 0.017 | 0.596 | 0.604 |
| SGD logistic | 0.790 ± 0.022 | 0.726 ± 0.043 | 0.790 ± 0.029 | 0.661 | 0.602 |
| Gaussian NB | 0.782 ± 0.030 | 0.782 ± 0.030 | 0.782 ± 0.029 | 0.589 | 0.589 |
| Hoeffding tree | 0.774 ± 0.035 | 0.782 ± 0.030 | 0.774 ± 0.034 | 0.589 | 0.590 |

AUC is the ranking number. The paired gap that matters is fractured minus two sessions, on the same test photos. Logistic is +0.004 ± 0.009. The small MLP is −0.011 ± 0.024 and the 128-wide net is −0.015 ± 0.036, both inside the seed noise. CatBoost is −0.029 ± 0.016, lower on all 10 seeds. LightGBM is −0.040 ± 0.023, lower on all 10. Sklearn boosted trees are −0.039 ± 0.031, lower on 9 of 10. SGD is −0.064 ± 0.049, lower on 8 of 10. The adaptive forest is −0.021 ± 0.025, lower on 8 of 10.

Logistic regression still matches the refit when the 400 labels arrive in seven pieces. Paired against replay the fractured AUC gap is +0.002 ± 0.010. The test AUC climbs 0.634, 0.679, 0.729, 0.753, 0.767, 0.778, 0.812 after 20, 40, 90, 120, 190, 240, and 400 labels. CatBoost is the best model when the two sessions are 200 photos each: AUC 0.820, delete F1 0.681, delete recall 0.742, log loss 0.519, against a replay of 0.818. Slicing the same photos into the seven batches costs that whole advantage. LightGBM tells the same story one step lower, 0.814 on two sessions and 0.773 fractured, and both of those two-session scores match their own replay.

The 128-wide net does not improve on 53→16→1. Two sessions are 0.811 against 0.814. Fractured, its log loss is 0.625 against 0.573 for the small net and 0.543 for its own replay, so the extra width is the one that drifts when the batches get short.

SGD on two sessions of 200 lands at AUC 0.790, the same as fitting it on all 400, with a worse log loss than the Hessian logistic (0.67 against 0.55). The seven-batch schedule is where it breaks: AUC 0.726 and log loss 1.37. The path gets to 0.735 after 240 labels and then the last batch of 160 pulls it down to 0.726.

Gaussian NB is unchanged by the schedule, which is what a sum of sufficient statistics should do. AUC 0.782, delete F1 0.589, and it calls delete on about 26% of the test photos. Log loss is 2.7. That repeats the earlier full-data result: a weak ranking and probabilities that are not usable.

The Hoeffding tree, with grace period 20 and delta 0.01, does not grow on this library. A pass over 400 rows of seed 0 left a single leaf. The library defaults are grace period 200 and delta 1e-7, which are for streams of millions; the relaxed settings still never authorized a split. The score is the naive Bayes classifier in that leaf, which is why it sits on top of Gaussian NB. Forcing splits, on that same 400-row pass, grew a 5-node tree and the AUC fell from 0.747 to about 0.70. The adaptive random forest is the online tree model that actually ranks. Ten trees, its own defaults, log loss about 0.53. At 20 labels it is the best of this set, AUC 0.684, ahead of logistic at 0.634. By 400 fractured labels it is at 0.775, and the two-session score is 0.796. Delete F1 stays near 0.60 because it only calls delete on about 30% of photos.

Sklearn boosted trees still want the rows kept. Two sessions reach 0.804 against a replay of 0.813. Seven batches fall to 0.765, and the path goes backwards when the second batch of 20 arrives: 0.647 after the first 20, 0.617 after 40. CatBoost dips there too, 0.639 then 0.629, then recovers to 0.791, which is still short of its two-session score.

For two large sessions, CatBoost continued from the first model is the strongest score in this round, and logistic regression plus the small MLP are tied with training on all 400. For a user whose sessions are short and uneven, the Hessian logistic is the update that still matches the refit. The boosters should be refit from stored feature rows.

## Follow-up: F1, class balance, learning rate, checkpoints

AUC and delete-class F1 answer different questions. AUC ranks by P(delete) and does not use a cutoff, which is what the swipe queue uses. F1 is precision and recall of the delete decision at the 0.5 cutoff. Always predicting keep scores accuracy 0.61, AUC 0.50, and delete F1 0. A model that only keeps has F1 0. The accuracy line in the first table was the always-keep baseline sitting next to the models. It was easy to read as "the model is no better than always keeping." Delete F1 is the number that shows the difference.

On the 53-feature schema, ten test folds, the epoch with the best validation log loss:

| training | model | AUC | delete F1 | delete recall | fraction predicted delete |
| --- | --- | ---: | ---: | ---: | ---: |
| none | always keep | 0.50 | 0.00 | 0.00 | 0.00 |
| unbalanced | logistic | 0.82 ± 0.03 | 0.65 | 0.62 | 0.35 |
| balanced | logistic | 0.82 ± 0.03 | 0.70 | 0.74 | 0.44 |
| unbalanced | boosted trees, lr 0.1 | 0.83 ± 0.03 | 0.63 | 0.58 | 0.33 |
| balanced | boosted trees, lr 0.1 | 0.83 ± 0.03 | 0.68 | 0.74 | 0.46 |
| unbalanced | MLP 53→16→1 | 0.82 ± 0.03 | 0.63 | 0.60 | 0.35 |
| balanced | MLP 53→16→1 | 0.82 ± 0.03 | 0.67 | 0.71 | 0.45 |
| unbalanced | MLP 53→64→32→1 | 0.82 ± 0.02 | 0.65 | 0.62 | 0.35 |
| balanced | MLP 53→64→32→1 | 0.83 ± 0.03 | 0.68 | 0.73 | 0.45 |

Class weighting leaves AUC in the same place and raises delete F1, because the model then calls delete on about 44% of photos instead of about 35%, and recall goes from about 0.60 to about 0.74. For a ranked queue, balance is optional. For a 0.5 keep/delete cutoff, balanced training is the better operating point.

Neither MLP collapsed to always-keep. A collapsed fold would have delete recall under 0.05. That happened in 0 of 10 folds at every learning rate from 0.0001 to 0.01, balanced and unbalanced, for both widths. The weights at the epoch where patience stops are in the same range, and often a little higher on test F1 than the validation-loss checkpoint. For the balanced big MLP that is F1 0.72 at the stopping epoch and 0.68 at the best validation epoch. The two nets look similar because both rank deletes, with recall around 0.6–0.7.

The learning-rate sweep is flat. MLP AUC stays inside about 0.80–0.83 from 0.0001 to 0.01. The plan's 0.001 is in the middle of that plateau. The weak end is the small net at 0.0001 unbalanced (F1 0.59, recall 0.53), which still predicts deletes. Boosted trees at learning rate 0.05, 0.1, and 0.3 stay inside AUC 0.82–0.83. The plan's 0.1 is the best of those three (0.833 unbalanced).

WhatsApp is constant on this export, and so are the two text slots, because text detection was not run. An unguarded z-score divides by that zero standard deviation and the whole vector becomes NaN. Three safe treatments produce the same logistic AUC and the same F1, to three decimals (0.821 and 0.70 when balanced):

- if the standard deviation is 0, use scale 1, so the column centers to 0
- force a flat column to 0 and standardize the rest
- drop the flat columns

Keeping WhatsApp is fine. Skipping standardization entirely also works for logistic regression (AUC 0.83) and costs the big MLP a little (AUC 0.81 versus 0.83 with the guarded scaler). The trees already use the raw vector. The implementation piece is the zero-standard-deviation guard. It is not a reason to remove the column.

Removing the GPS flag, which the current plan does, drops balanced logistic AUC from 0.84 to 0.82 and unbalanced tree AUC from 0.84 to 0.83. Balanced logistic F1 stays 0.70. The first table on this page still includes GPS.

## What to ship

Logistic regression, the small boosted trees, and the small MLP all land in the same place. The big MLP does not improve on the small one. A calibrated linear SVM matches logistic regression, so leaving SVM out is the right call.

| model | AUC | log loss | accuracy | top-20% precision |
| --- | ---: | ---: | ---: | ---: |
| always guess the base rate | 0.50 | 0.67 | 0.61 | 0.37 |
| screenshot rule | 0.56 | 0.64 | 0.65 | 0.50 |
| logistic regression, L2 = 1, balanced | 0.84 ± 0.03 | 0.51 | 0.75 | 0.84 |
| gradient-boosted trees, clean | 0.84 ± 0.03 | 0.47 | 0.76 | 0.86 |
| MLP 59→16→1, clean | 0.86 ± 0.02 | 0.47 | 0.77 | 0.85 |
| MLP 59→64→32→1, clean | 0.86 ± 0.02 | 0.49 | 0.77 | 0.85 |

Accuracy of 0.76 compares with 0.61 for always predicting keep. The front of the queue (the highest 20% of P(delete)) is about 85% true deletes, against a 40% base rate, and that slice covers about 42% of the deletes.

On this 59-feature run the big MLP matches the small one on AUC (0.857 vs 0.856) with a worse log loss (0.490 vs 0.474). The follow-up on the current 53-feature schema, using the best validation checkpoint, does not show a training collapse: both nets predict delete on about 35–45% of the test photos, and the big net is a small step ahead (AUC 0.83 vs 0.82, delete F1 0.68 vs 0.67). At 20–80 labels the big net was the weakest. Extra width is optional.

Keep logistic regression as the default. It is the stable one: train-to-test AUC gap about 0.05, and it is the only model that still ranks when the label set is small. Keep the boosted trees. They match logistic regression and have the best log loss of the simple models (0.47), with a larger train-to-test AUC gap (0.12). Keep the small MLP. It is tied with the trees on log loss and a little higher on AUC, inside the noise of ±0.02.

These did not beat that set, so there is no reason to add them: histogram gradient boosting (AUC 0.85), a random forest (0.85), and a calibrated linear SVM (0.84, log loss 0.50). Gaussian NB produced a log loss of 2.9. The plan's decision to skip SVM is confirmed by the numbers.

## The feature vector works

Photometric stats alone reach AUC 0.75. YOLO category scores alone reach 0.76–0.77. YuNet faces alone reach 0.71. Metadata alone reaches 0.70–0.76. The full 59-d vector reaches 0.84. Removing metadata drops logistic regression to 0.82. Removing faces drops it to 0.82. The pieces add up, and no single block is the whole model.

Shuffling a group on the test fold drops logistic AUC by:

- faces, 0.09
- YOLO categories, 0.08
- metadata, 0.06
- photometric stats, 0.04
- YOLO geometry, about 0
- the screenshot bit alone, about 0

The strongest single columns, read as "higher means keep": max detection confidence (strength 0.74), luminance entropy (0.73), max face confidence (0.71), person max confidence (0.71), has-GPS (0.69).

Photos with a YuNet face were deleted 18% of the time. Photos without one were deleted 55% of the time. Photos with a GPS flag were deleted 25% of the time; photos without one, 61%. That GPS gap is still 25% vs 57% after screenshots are removed, so the flag is a real keep signal on this library, which is what the plan's yes/no GPS bit is for. Coordinates stay out.

Screenshots are 39 photos, and 34 of them are deletes (87%). As a rule by itself that is only AUC 0.56, because 39 photos cannot move the whole ranking. The filename rule and the catalog flag agree on all 39, so the screenshot bit in the vector is the same one the app would compute. Inside the full model, a screenshot that also contains a person is often scored as a keep. On the first test split, three of the eight lowest P(delete) deletes were PNGs with a person box around 0.9. The face signal outweighs the screenshot bit. With only 8 screenshots that contain a face (6 of them deletes), this is a small failure mode, and it is the one to watch when Vision is wired up.

## Schema choices that held up

**Category buckets vs 80 COCO classes.** Replacing the 22 category numbers with 80 per-class max confidences did not help. Categories alone: logistic AUC 0.76, log loss 0.58. The 80-d vector: AUC 0.77, log loss 0.67. A full vector that uses the 80 classes instead of the buckets is slightly worse for logistic regression (AUC 0.83, log loss 0.61) and tied for the trees (AUC 0.84). Keep the 11 buckets.

**Empty and rare categories can stay.** Appliance, sports, and outdoor each appear in under 5% of photos, and for those the sum of confidences is almost the same number as the max (correlation 0.95–0.99). Person and kitchen are different: 42% and 62% of the photos that contain them contain more than one box, and max-vs-sum correlation drops to 0.68 and 0.78. Dropping the eight columns that are entirely constant did not change AUC. The plan's claim that L2, trees, and early stopping can ignore an empty column holds on this library. Those columns should stay in the schema, because another library will have WhatsApp photos and Live Photos even though this export does not.

**Confidence threshold and input size.** At 0.10, 0.25, and 0.50, AUC stays inside 0.843–0.848. A 416 input (the plan's YOLOX letterbox) matches a 640 input (YOLO11n's native size): logistic AUC 0.848 vs 0.843. The plan's 0.25 and 416 are fine. This is the part most likely to shift a little when the real YOLOX-Nano Core ML model replaces YOLO11n. The boxes on spot-checked photos were on the right objects (people, cars, a group), so the aggregation itself is doing the right thing.

**Geometry is redundant once categories exist.** The six geometry numbers look useful on their own (AUC 0.72) because "anything was detected" means keep. After the category scores are present, shuffling geometry changes AUC by about 0. The six numbers are cheap and can stay. They are not where the accuracy comes from.

## Issues to fix in the implementation

**The scaler has to tolerate a zero standard deviation.** On this export WhatsApp and the two text slots are constant, so an unguarded z-score is NaN. Setting the scale to 1 when the standard deviation is 0, or writing a flat column as 0, matches dropping those columns exactly (logistic AUC 0.821 either way). WhatsApp stays in the schema. The follow-up section has the comparison. The trees do not standardize.

**A cap of 15 IRLS iterations never finished.** Every split used all 15 steps (`converged` was 0). Running until convergence took about 41 iterations. AUC was unchanged (0.843). Log loss improved from 0.520 to 0.511. Ranking is fine at 15. The log loss the UI shows gets a little better if the solver is allowed to finish. A cap around 50 covers this dataset.

**L2 strength 1 is fine.** C = 0.1 (stronger regularization) reached AUC 0.850 and log loss 0.480. C = 1 reached 0.843 and 0.511. C = 10 was slightly worse. The gap is inside the ±0.03 split noise. On the current 53-feature schema, class weighting leaves AUC unchanged and raises delete F1 (logistic 0.70 balanced versus 0.65 unbalanced). See the follow-up table.

**Picking the saved round on the test split flatters the on-screen loss by a small amount.** For the trees, test-split selection reported log loss 0.467 versus an honest 0.473, and it kept about 36 of the 40 trees. For the small MLP the gap was 0.459 versus 0.474 (best epoch around 32). For the big MLP it was 0.466 versus 0.490 (best epoch around 10). The split noise on log loss is about 0.03–0.07, so this does not change which model wins. It does mean the number in the training chart is a little optimistic, and more so for the MLP. An inner validation slice is the clean place to choose the round. The chart can still draw the test curve.

**The sharpness ratio can spike.** Median center/border Laplacian ratio is 1.3, the 99th percentile is 15, and the maximum is 2410. Four photos in a thousand exceed 100, which is a flat border divided by a tiny epsilon (`1e-8`). Replacing that column, the Laplacian variance, and the noise MAD with `log1p` moved logistic AUC from 0.843 to 0.848 and left the trees unchanged, because a tree split is invariant to a monotone rescaling. A cap or `log1p` on the ratio keeps one flat-border photo from setting the standard deviation of that column. Low urgency.

**The four luminance-quarter masses sum to 1**, so they are an exact linear dependence, and several max/sum pairs are near copies. That is why the condition number of the standardized matrix is enormous. L2 is what keeps logistic regression well behaved on it. The histogram block can stay. Specifying the formulas matters more than dropping a column: Rec.709 luminance, 32-bin entropy, central half of each side for the sharpness ratio, MAD of luminance minus a 3×3 box blur for the noise term, and natural log for the counts. Train and predict have to share those choices. The plan leaves them open.

**The minimum of 20 labels is too low for the number the UI will show.** A 20-photo set has a test fold of about 4 photos, and AUC there swings by ±0.2 to ±0.3. At 20 labels the big MLP was at chance (AUC 0.50) and the trees were at 0.55. Logistic regression's average looked higher and its standard deviation was 0.32, so a single run is not a result. By about 300 labels, logistic regression, the trees, and both MLPs sit near their full-data scores. The gate of 20 with 2 per class can stay as a hard error. The models worth offering the user start earning their keep somewhere around 150–300 labels, and logistic regression is the one to run if the user trains earlier than that.

## Signals the plan leaves out

Creation time is a weak keep signal (strength 0.59: newer photos are kept a bit more often). Leaving it out is the right call, and on this library it costs little.

File size and megapixels are stronger (strength about 0.66–0.67, larger means keep) and they remain after screenshots are removed (0.65 and 0.63). They mostly say "this is a full-resolution camera photo." GPS, faces, and the photometric stats already carry that. I would leave file size out. It also tracks format and recompression.

## Scope

These scores are on photos this user already marked keep or delete, 685 out of about 13,000 in the catalog. That is the training population. It is a biased slice of the library.

YOLO11n is not YOLOX-Nano. Same 80 classes, different head, and the 416-vs-640 comparison is the check on the plan's letterbox size. Text detection was not run, so the two text slots are untested and should stay in the schema for Vision. The face result is from YuNet at a score threshold of 0.5, which the plan does not specify.
