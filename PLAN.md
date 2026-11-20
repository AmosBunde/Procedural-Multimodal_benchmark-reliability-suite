# Plan

Amos Bunde, 20 November 2026

Written before any model was scored on the study suites. The section "What I had seen
when I wrote this" says exactly what had been run by then. Anything decided later goes
under "Amendments" at the end, with a date.

## Question

A benchmark whose items are generated from a seed can be thrown away and regenerated, so
it cannot leak into training data and cannot be memorised. That is only useful if a
regenerated copy measures the same thing as the copy it replaces. So the question is about
the benchmark, not about the models:

**Can a procedurally generated image and audio benchmark be trusted to rank models, and
what does it take to know?**

I answer it by building the benchmark, running real models on it, and then measuring the
benchmark with the same care one would give a model: ranking stability, required size,
reproducibility across regenerated copies, sensitivity to caption wording, item quality,
floor and ceiling effects, and redundancy between tasks.

## What is already known

- Contrastive image-text models are weak at counting, at binding attributes to objects and
  at spatial relations. Winoground (Thrush et al., CVPR 2022) and ARO (Yuksekgonul et al.,
  ICLR 2023) established this on natural images. I expect to see it again and claim
  nothing new if I do.
- Procedurally generated diagnostic benchmarks with exact ground truth are an old idea;
  CLEVR (Johnson et al., CVPR 2017) is the standard example.
- Classical test theory (item difficulty, item discrimination, KR-20) is decades old.

What this project adds is the reliability measurement applied to a regenerable benchmark,
and the code to repeat it on other models and other suites.

## Design

### Benchmark

Every item is one image or one audio clip with two or four candidate captions, exactly one
of which is true. The item stores the scene it was rendered from, so the truth can be
checked from the scene and from the pixels or samples. Each task has three difficulty
levels with equal numbers of items.

Image tasks (224 x 224 pixels, flat coloured shapes on a grey background):

| Task | Question | Choices | What the levels change |
|---|---|---|---|
| object | which colour and shape (control task) | 4 | object size, position, pixel noise |
| count | how many shapes of one kind | 4 consecutive counts | count range (1-4, 2-7, 3-9), size, distractors, noise |
| binding | which colour goes with which shape | 2 (true, colours swapped) | size, distractors, noise |
| spatial | left of, right of, above or below | 2 (true, opposite) | separation, misalignment, distractors, noise |
| size | which of two same-shaped objects is larger | 2 (true, swapped) | size ratio (2.0-2.6, 1.5-1.8, 1.25-1.4), noise |
| digits | read a rendered number | 4 | digits (2, 3, 4), font size, near-miss foils, contrast, noise |

Audio tasks (10 seconds at 48 kHz, synthesised tones, noise bursts and clicks):

| Task | Question | Choices | What the levels change |
|---|---|---|---|
| bursts | how many beeps | 4 consecutive counts | count range, burst length, gaps, noise |
| pitch | does a tone rise or fall | 2 | size of the sweep (2 octaves down to 15-30%), noise |
| order | beep then noise, or noise then beep | 2 | event length and gap, noise |
| tempo | do clicks speed up or slow down | 2 | how much the spacing changes, noise |

Each task has three caption templates: a canonical wording and two paraphrases. Each item
also has a plain question-and-answer rendering for generative models. No generative model
is evaluated in this study.

### Models

All are public checkpoints downloaded anonymously from Hugging Face and run on a CPU.

| Key | Checkpoint | Loader |
|---|---|---|
| openai-rn50 | `timm/resnet50_clip.openai` | open_clip |
| openai-vit-b32 | `timm/vit_base_patch32_clip_224.openai` | open_clip |
| laion2b-vit-b32 | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | open_clip |
| datacomp-vit-b32 | `laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K` | open_clip |
| metaclip-vit-b32 | `timm/vit_base_patch32_clip_224.metaclip_400m` | open_clip |
| siglip-vit-b16 | `timm/ViT-B-16-SigLIP` | open_clip |
| clap-htsat-unfused | `laion/clap-htsat-unfused` | transformers |
| clap-larger-general | `laion/larger_clap_general` | transformers |
| clap-larger-music-speech | `laion/larger_clap_music_and_speech` | transformers |

Six image-text models from five training sources and three audio-text models. A model
answers an item by scoring each candidate caption against the media; the item is correct
if the true caption scores strictly highest.

### Sample

The machine is a four-core laptop without a GPU, and I allow about an hour of model time
in total. That fixes the sample:

- **Images:** 5 suites from seeds 0 to 4, each 72 items per task (24 per level), so 432
  items per suite and 2,160 in all (360 per task). Seed 0 carries all three caption
  templates; seeds 1 to 4 carry the canonical template only, because text encoding is
  the slow step for one of the models.
- **Audio:** 3 suites from seeds 0 to 2, each 30 items per task (10 per level), so 120
  clips per suite and 360 in all (90 per task), all with three templates.

What this size can and cannot show. With 360 items a task accuracy near 50% has a 95%
interval of about plus or minus 5 points, and with 90 items about plus or minus 10. Two
models whose overall scores differ by less than about 2 points will not be separated. Six
models give 15 pairs, which is enough to ask whether a ranking is stable and far too few
to say anything general about model families. Three audio models give three pairs; the
audio analysis is a demonstration that the pipeline and the checks carry over to a second
modality, not a ranking study.

### Analysis

The overall score of a model is the unweighted mean of its task accuracies under the
canonical template. "Pooled" means all suites of a modality together.

1. **Ranking stability.** Stratified bootstrap over items (resampled within task, the same
   resample for every model, 2,000 resamples): 95% percentile intervals on scores, on
   ranks, and on every pairwise difference. A pair is "separated" if its interval excludes
   zero. Exact McNemar tests with Holm correction are reported beside it.
2. **Items needed.** Draw two disjoint random sets of n items per task from the pool and
   compare the rankings they give (1,000 draws per n): Kendall tau, same top model, same
   top three, same complete order.
3. **Seed reproducibility.** Score every model on each regenerated suite. Report the
   standard deviation of the overall score across seeds beside the value binomial sampling
   predicts, Kendall tau between the rankings of every pair of suites, and a chi-square
   test of homogeneity across seeds for every model-task cell.
4. **Template sensitivity.** On the seed-0 suite, score every model under each template.
   Report the spread of the overall score across templates, rank agreement between
   templates, the share of items whose outcome changes with the wording, and paired
   bootstrap intervals for the change from the canonical wording.
5. **Item analysis.** With models as examinees: the share of models that get each item
   right, and the correlation between getting the item right and the model's overall score.
   Flag items no model gets right, items every model gets right, and items with a
   correlation of -0.4 or below. Because six models make that correlation very noisy, the
   number of flagged items is compared with an exact count of how many would be flagged if
   the same number of models got each item right at random. KR-20 and split-half
   reliability per task. No IRT model is fitted: six examinees cannot support one.
6. **Floor and ceiling.** A model is above chance on a task if an exact one-sided binomial
   test against the task's chance level gives p < 0.05 after Holm correction over models.
   A task is **at floor** if no model is above chance, and **at ceiling** if every model
   scores 95% or more. Otherwise it is "informative".
7. **Redundancy.** Spearman correlation between tasks of the models' accuracies, with
   bootstrap intervals, and the effect on the overall ranking of leaving each task out.

Random numbers in the analysis come from fixed seeds, so `summarise` is deterministic.

## Hypotheses

Each has a rule that is applied by code (`src/regenbench/hypotheses.py`). A hypothesis
with several parts is supported only if all parts hold, failed if none holds, and partly
supported otherwise.

**H1. A regenerated suite reproduces the result.** Across the five image suites: (a) mean
Kendall tau between the rankings of every pair of suites is at least 0.70; (b) the same
model is top in at least four of five suites; (c) the homogeneity test rejects at the 5%
level in at most 10% of model-task cells, which is what independent draws from one
generator should give.

**H2. Wording matters more than regeneration.** For at least four of the six image
models, the standard deviation of the overall score across the three templates (seed 0) is
larger than its standard deviation across the five seeds (canonical template).

**H3. Wording matters as much as the model.** (a) The range of the overall score across
templates, averaged over models, is larger than the median gap between neighbouring models
in the pooled ranking; (b) at least one pair of templates orders the models differently.

**H4. Some tasks cannot rank anything.** At least two of the six image tasks are at floor
on the pooled items. I expect them to be `spatial` and `binding`; each named task counts
as a part.

**H5. The control task saturates.** On the level-1 `object` items (120 pooled), every
model scores 95% or more.

**H6. One suite fixes the top of the ranking but not the order.** Two independent suites
of 72 items per task agree on the top model in at least 90% of draws and on the top three
in at least 80%. But even at the largest size that can be tested (180 items per task) two
suites agree on the complete order of six models in fewer than 50% of draws.

**H7. Items in dead tasks carry no signal.** The mean item-ability correlation is within
0.05 of zero for every task at floor, and above 0.10 for every informative task. Not
testable if no task is at floor.

**H8. The difficulty knob works.** In every image task that is not at floor, accuracy
averaged over models falls from level 1 to level 2 and from level 2 to level 3.

**H9. Audio.** At least two of the four audio tasks are at floor on the pooled clips. I
expect `order` and `tempo`; each named task counts as a part. This rests on three models
and 90 clips per task and is exploratory.

My confidence, for the record: high for H4 and H5, moderate for H1, H3, H6 and H8, low for
H2, H7 and H9.

## What I had seen when I wrote this

- **No accuracy of any real model on any item.** I have not computed or looked at one.
- **Timing pilots.** I timed the image and text encoders of each model on random tensors
  and on a few fixed captions, and ran all nine models through the pipeline on a pilot
  suite from seed 999 (54 images, 24 clips) to measure seconds per item. I checked only
  that the scores were finite and had the right shape. The pilot result files were not
  opened and have been deleted. Seed 999 is not used in the study.
- **A synthetic run.** To develop the analysis, the figures and the report I ran nine fake
  "models" that score captions with a hash, under the names of the real models. They are
  at chance by construction and say nothing about the real ones.
- **Design changes forced by the timing pilots.** The first design had seven or eight
  image models including OpenAI ViT-B/16, ViT-L/14 and LAION ViT-B/16, and 120 items per
  task per suite. The laptop throttles under sustained load and ran at roughly half the
  speed I first measured, so I removed the three largest CLIP models, kept SigLIP ViT-B/16
  as the one non-CLIP image model, cut the image suites to 72 items per task and the audio
  suites to 30. These choices were made on speed alone.
- **A text-encoding shortcut.** For CLIP models with a causal text tower I skip the
  padding positions after the end-of-text token, which cannot affect the result. The
  adapter checks at load time that the shortcut reproduces the full-length embedding and
  disables it otherwise. On the pilot captions the largest difference was a few parts in a
  million.
- **The literature above**, which is why H4 names `spatial` and `binding`.

## Limits known in advance

- Zero-shot caption matching is not how generative multimodal language models are used.
  The suite has a question-and-answer format for them, but none is evaluated here.
- Flat synthetic shapes and synthetic tones are far from the photographs, captions and
  recordings these models were trained on. Low accuracy may reflect that distance and not
  a missing skill.
- Six image models, four of them ViT-B/32. Item statistics and task correlations computed
  over six examinees are noisy, and I report them with that caveat.
- The plan is not registered anywhere, and its author runs the experiments.

## Amendments

None yet.
