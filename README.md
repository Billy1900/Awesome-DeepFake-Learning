# Awesome-DeepFake-Learning
The approach I work on DeepFake.

- [Awesome-DeepFake-Learning](#awesome-deepfake-learning)
  * [1. Intuitive Learning](#1-intuitive-learning)
  * [2. Survey Paper](#2-survey-paper)
  * [3. Curated lists](#3-curated-lists)
  * [4. Deepfakes Datasets](#4-deepfakes-datasets)
  * [5. Generation of synthetic content](#5-generation-of-synthetic-content)
    + [5.1 Generation Text](#51-generation-text)
      - [⚒️ Tools ⚒️](#---tools---)
      - [📃 Papers 📃](#---papers---)
      - [🌐 Webs 🌐](#---webs---)
      - [😎 Awesome 😎](#---awesome---)
    + [5.2 Generation Audio](#52-generation-audio)
      - [⚒️ Tools ⚒️](#---tools----1)
      - [📃 Papers 📃](#---papers----1)
    + [5.3 Generation Images](#53-generation-images)
      - [⚒️ Tools ⚒️](#---tools----2)
      - [📃 Papers 📃](#---papers----2)
      - [🌐 Webs 🌐](#---webs----1)
      - [😎 Awesome 😎](#---awesome----1)
    + [5.4 Generation Videos](#54-generation-videos)
      - [⚒️ Tools ⚒️](#---tools----3)
      - [📃 Papers 📃](#---papers----3)
      - [🌐 Webs 🌐](#---webs----2)
      - [📺 Videos 📺](#---videos---)
  * [6. Detection of synthetic content](#6-detection-of-synthetic-content)
    + [6.1 Detection Text](#61-detection-text)
      - [⚒️ Tools ⚒️](#---tools----4)
      - [📃 Papers 📃](#---papers----4)
    + [6.2 Detection Audio](#62-detection-audio)
      - [⚒️ Tools ⚒️](#---tools----5)
      - [📃 Papers 📃](#---papers----5)
    + [6.3 Detection Images](#63-detection-images)
      - [⚒️ Tools ⚒️](#---tools----6)
      - [📃 Papers 📃](#---papers----6)
    + [6.4 Detection Videos](#64-detection-videos)
      - [⚒️ Tools ⚒️](#---tools----7)
      - [📃 Papers 📃](#---papers----7)
      - [📺 Videos 📺](#---videos----1)
      - [😎 Awesome 😎](#---awesome----2)
  * [7. Misc](#7-misc)
    + [Articles](#articles)
    + [Talks](#talks)
    + [Challenges](#challenges)
    + [Forums](#forums)


## 1. Intuitive Learning
- [News From CNN](https://edition.cnn.com/interactive/2019/01/business/pentagons-race-against-deepfakes/)
- [10 DeepFake Examples](https://www.creativebloq.com/features/deepfake-examples)
- [An Introduction to DeepFakes](https://www.alanzucconi.com/2018/03/14/introduction-to-deepfakes/)
- 荷兰初创公司[Deeptrace](https://deeptracelabs.com/)发布两个年度（2018年和2019年）[Deepfakes发展状况调研报告](https://github.com/Qingcsai/Deepfakes-Zoo/tree/master/the-state-of-deepfakes)
And I find a very interesting [**website**](http://www.seeprettyface.com/index.html) which could generate human face via Deepfake tech and it makes its code open source which could help us learn.
## 2. Survey Paper
- [The Creation and Detection of Deepfakes: A Survey](https://arxiv.org/abs/2004.11138)
- [Media Forensics and DeepFakes: an overview](https://arxiv.org/abs/2001.06564)
- [DeepFakes and Beyond: A Survey of Face Manipulation and Fake Detection](https://arxiv.org/abs/2001.00179)
- [Deep Learning for Deepfakes Creation and Detection: A Survey](https://arxiv.org/abs/1909.11573)

## 3. Curated lists
- [Deep-Learning-for-Tracking-and-Detection](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection#synthetic_gradient_s_): Collection of papers, datasets, code and other resources for object tracking and detection using deep learning

## 4. Deepfakes Datasets

Datasets|Year|Ratio<br>tampered:original|Total videos|Source|Participants Consent|Tools
:-------:|:----:|:-----------:|:----:|:---:|:-----:|:--:
[UADFV](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8630787)|2018|1 : 1.00|98|YouTube|N|FakeAPP
[FaceForensics](https://arxiv.org/abs/1803.09179)|2018|1 : 1.00|2008|YouTube|N|Face2Face
[Deepfake-TIMIT](https://www.idiap.ch/dataset/deepfaketimit)|2019|1 : 1.00|620|Vid-TIMIT|N|faceswap-GAN 
[FaceForensics++](https://github.com/ondyari/FaceForensics)|2019|1 : 0.25|5000|YouTube|N|faceswap <br> DeepFake <br> Face2Face <br> NeuralTextures
[DeepFakeDetection<br>(part of FaceForensics++)](https://deepfakedetectionchallenge.ai/dataset)|2019|1 : 0.12|3363|Actors|Y
[Celeb-DF](http://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html)|2019|1 : 0.51|1203|YouTube|N|a refined version of the DeepFake
[DFDC Preview Dataset](https://deepfakedetectionchallenge.ai/dataset)|2019|1 : 0.28|5214|Actors|Y|Unkonwn


## 5. Generation of synthetic content

### 5.1 Generation Text

#### ⚒️ Tools ⚒️

| Name | Description | Demo | Popularity |
| ---------- | :---------- | :---------- | :----------: |
| [Grover](https://github.com/rowanz/grover) | Grover is a model for Neural Fake News -- both generation and detection. However, it probably can also be used for other generation tasks. | [https://grover.allenai.org/](https://grover.allenai.org/) | [![stars](https://badgen.net/github/stars/rowanz/grover)](https://github.com/rowanz/grover)|
[gpt-2xy](https://github.com/NaxAlpha/gpt-2xy) | GPT-2 User Interface based on HuggingFace's Pytorch Implementation | https://gpt2.ai-demo.xyz/ | [![stars](https://badgen.net/github/stars/NaxAlpha/gpt-2xy)](https://github.com/NaxAlpha/gpt-2xy) |
[CTRL](https://github.com/salesforce/ctrl) | Conditional Transformer Language Model for Controllable Generation | N/A | [![stars](https://badgen.net/github/stars/salesforce/ctrl)](https://github.com/salesforce/ctrl) |
[Talk to Transformer](https://talktotransformer.com/) | See how a modern neural network completes your text. Type a custom snippet or try one of the examples | https://talktotransformer.com | N/A |
[LEO](http://leoia.es) | First intelligent system for creating news in Spanish | N/A | N/A
[Big Bird](https://bigbird.dev/) | Bird Bird uses State of the Art (SOTA) Natural Language Processing to aid your fact-checked and substantive content. | [BigBirdDemo](https://run.bigbird.dev/auth/login)| N/A
[aitextgen](https://github.com/minimaxir/aitextgen) | A robust Python tool for text-based AI training and generation using GPT-2.| [Demo](https://git.io/JfuXR) | [![stars](https://badgen.net/github/stars/minimaxir/aitextgen)](https://github.com/minimaxir/aitextgen)
[GPT-3](https://github.com/openai/gpt-3) | GPT-3: Language Models are Few-Shot Learners | N/A | [![stars](https://badgen.net/github/stars/openai/gpt-3)](https://github.com/openai/gpt-3)

#### 📃 Papers 📃

* [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
* [Saliency Maps Generation for Automatic Text Summarization](https://arxiv.org/pdf/1907.05664.pdf)
* [Automatic Conditional Generation of Personalized Social Media Short Texts](https://arxiv.org/pdf/1906.09324.pdf)
* [Neural Text Generation in Stories Using Entity Representations as Context](https://homes.cs.washington.edu/~eaclark7/www/naacl2018.pdf)
* [DeepTingle](https://arxiv.org/pdf/1705.03557.pdf)
* [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)
* [Evaluation of Text Generation: A Survey](https://arxiv.org/pdf/2006.14799.pdf)

#### 🌐 Webs 🌐

* [NotRealNews](https://notrealnews.net/)
* [BotPoet](http://botpoet.com/vote/sign-post/)
* [TheseLyricsDoNotExist](https://theselyricsdonotexist.com/)
* [ThisResumeDoesNotExist](https://thisresumedoesnotexist.com/)
* [NotRealNews](https://notrealnews.net/)
* [ThisArtWorkDoesnotExist](https://thisartworkdoesnotexist.com/)
* [BoredHumans](https://boredhumans.com/quotes.php)
* [GPT-2 Neural Network Poetry](https://www.gwern.net/GPT-2)
* [A.ttent.io](https://a.ttent.io/n/?v=7)
* [ThisEpisodeDoesNotExist](https://thisepisodedoesnotexist.com)

#### 😎 Awesome 😎

* [awesome-text-generation](https://github.com/ChenChengKuan/awesome-text-generation)
* [Awesome GPT-3](https://github.com/elyase/awesome-gpt3)

### 5.2 Generation Audio

#### ⚒️ Tools ⚒️

| Name | Description | Demo | Popularity |
| ---------- | :---------- | :---------- | :----------: |
[Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) | Clone a voice in 5 seconds to generate arbitrary speech in real-time | https://www.youtube.com/watch?v=-O_hYhToKoA | [![stars](https://badgen.net/github/stars/CorentinJ/Real-Time-Voice-Cloning)](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
[Lyrebird](https://beta.myvoice.lyrebird.ai/) | Create your own vocal avatar! | N/A | N/A |
[Descrypt](https://www.descript.com/) | Record. Transcribe. Edit. Mix. As easy as typing. | N/A | N/A
[Common Voice](https://voice.mozilla.org/en) | Common Voice is Mozilla's initiative to help teach machines how real people speak. | N/A | N/A
[Resemble.ai](https://www.resemble.ai/) | Resemble can clone any voice so it sounds like a real human. | N/A | N/A
[TacoTron](https://google.github.io/tacotron/) | Tacotron (/täkōˌträn/): An end-to-end speech synthesis system by Google. | [Demo](https://google.github.io/tacotron/publications/prosody_prior/index.html) | [![stars](https://badgen.net/github/stars/google/tacotron)](https://github.com/google/tacotron)
[Sonantic](https://sonantic.io/) | Create a captivating performance using emotionally expressive text-to-speech. | [Demo](https://www.youtube.com/watch?v=WzBimNSO-U8) | N/A
[15.ai](https://fifteen.ai/) | Natural text-to-speech synthesis with minimal data. | [Demo](https://fifteen.ai/examples) | N/A


#### 📃 Papers 📃

* [Neural Voice Cloning with a Few Samples](http://research.baidu.com/Blog/index-view?id=81)
* [Data Efficient Voice Cloning for Neural Singing Synthesis](https://mtg.github.io/singing-synthesis-demos/voice-cloning/)
* [Efficient Neural Audio Synthesis](https://arxiv.org/pdf/1802.08435v1.pdf)
* [Score and Lyrics-free Singing Voice Generation](https://arxiv.org/pdf/1912.11747.pdf)
* [Generating diverse and natural Text-to-Speech samples using a quantized fine-grained vae and autoregressive prosody prior](https://arxiv.org/pdf/2002.03788.pdf)
* [Rave.dj](https://rave.dj/)

### 5.3 Generation Images

#### ⚒️ Tools ⚒️

| Name | Description | Demo | Popularity |
| ---------- | :---------- | :---------- | :----------: |
[StyleGAN](https://github.com/NVlabs/stylegan) | An alternative generator architecture for generative adversarial networks, borrowing from style transfer literature. The new architecture leads to an automatically learned, unsupervised separation of high-level attributes (e.g., pose and identity when trained on human faces) and stochastic variation in the generated images (e.g., freckles, hair), and it enables intuitive, scale-specific control of the synthesis. The new generator improves the state-of-the-art in terms of traditional distribution quality metrics, leads to demonstrably better interpolation properties, and also better disentangles the latent factors of variation. | https://www.youtube.com/watch?v=kSLJriaOumA | [![stars](https://badgen.net/github/stars/NVlabs/stylegan)](https://github.com/NVlabs/stylegan)
[StyleGAN2](https://github.com/NVlabs/stylegan2) | Improved version for StyleGAN. | https://www.youtube.com/watch?v=c-NJtV9Jvp0 | [![stars](https://badgen.net/github/stars/NVlabs/stylegan2)](https://github.com/NVlabs/stylegan2)
| [DG-Net](https://github.com/NVlabs/DG-Net) | Joint Discriminative and Generative Learning for Person Re-identification | https://www.youtube.com/watch?v=ubCrEAIpQs4 | [![stars](https://badgen.net/github/stars/NVlabs/DG-Net)](https://github.com/NVlabs/DG-Net)
| [GANSpace](https://github.com/harskish/ganspace) | Discovering Interpretable GAN Controls | http://www.exploreganspace.com/ | [![stars](https://badgen.net/github/stars/harskish/ganspace)](https://github.com/harskish/ganspace)
|[StarGAN v2](https://github.com/clovaai/stargan-v2) | StarGAN v2 - Official PyTorch Implementation (CVPR 2020) | https://youtu.be/0EVh5Ki4dIY | [![stars](https://badgen.net/github/stars/clovaai/stargan-v2)](https://github.com/clovaai/stargan-v2)
[Image GPT](https://github.com/openai/image-gpt) | Image GPT | N/A | [![stars](https://badgen.net/github/stars/openai/image-gpt)](https://github.com/openai/image-gpt)
[FQ-GAN](https://github.com/YangNaruto/FQ-GAN) | Official implementation of FQ-GAN | http://40.71.23.172:8888 | [![stars](https://badgen.net/github/stars/YangNaruto/FQ-GAN)](https://github.com/YangNaruto/FQ-GAN)
|[EHM_Faces](https://github.com/colinrsmall/ehm_faces) | EHM_Faces is a machine learning project that can generate high-quality, realistic ice hockey player portraits. Primarily meant for the game Eastside Hockey Manager (EHM), this project can generate portraits either one-at-a-time or in batches (the resulting batches are called facepacks). | N/A | [![stars](https://badgen.net/github/stars/colinrsmall/ehm_faces)](https://github.com/colinrsmall/ehm_faces)
|[Rewriting a Deep Generative Model](https://github.com/davidbau/rewriting) | Edits the weights of a deep generative network by rewriting associative memory directly, without training data | [Demo](https://colab.research.google.com/github/davidbau/rewriting/blob/master/notebooks/rewriting-interface.ipynb) | [![stars](https://badgen.net/github/stars/davidbau/rewriting)](https://github.com/davidbau/rewriting)


#### 📃 Papers 📃

* [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/pdf/1812.04948.pdf)
* [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/pdf/1912.04958.pdf)
* [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://arxiv.org/pdf/1711.11585.pdf)
* [Complement Face Forensic Detection and Localization with Facial Landmarks](https://arxiv.org/pdf/1910.05455.pdf)
* [Joint Discriminative and Generative Learning for Person Re-identification](https://arxiv.org/pdf/1904.07223.pdf)
* [Image2StyleGAN++: How to Edit the Embedded Images?](https://arxiv.org/pdf/1911.11544.pdf)
* [StyleGAN2 Distillation for Feed-forward Image Manipulation](https://arxiv.org/pdf/2003.03581.pdf)
* [Generative Pretraining from Pixels](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf)
* [Intrinsic Autoencoders for Joint Neural Rendering and Intrinsic Image Decomposition](https://arxiv.org/pdf/2006.16011.pdf)
* [StarGAN v2: Diverse Image Synthesis for Multiple Domains](https://arxiv.org/pdf/1912.01865.pdf)
* [Feature Quantization Improves GAN Training](https://arxiv.org/pdf/2004.02088.pdf)
* [High-Resolution Neural Face Swapping for Visual Effects](https://s3.amazonaws.com/disney-research-data/wp-content/uploads/2020/06/18013325/High-Resolution-Neural-Face-Swapping-for-Visual-Effects.pdf)
* [Improving Style-Content Disentanglement in Image-to-Image Translation](https://arxiv.org/pdf/2007.04964.pdf)
* [Rewriting a Deep Generative Model](https://arxiv.org/pdf/2007.15646.pdf)

#### 🌐 Webs 🌐

* [ThisPersonDoesNotExist](http://www.thispersondoesnotexist.com/)
* [WhichFaceIsReal](http://www.whichfaceisreal.com/)
* [ThisRentalDoesNotExist](https://thisrentaldoesnotexist.com/)
* [ThisCatDoesNotExist](https://thiscatdoesnotexist.com/)
* [ThisWaifuDoesNotExist](https://www.thiswaifudoesnotexist.net/)
* [thispersondoesnotexist](http://www.thispersondoesnotexist.com/)

#### 😎 Awesome 😎

* [Awesome Pretrained StyleGAN2](https://github.com/justinpinkney/awesome-pretrained-stylegan2)

### 5.4 Generation Videos

#### ⚒️ Tools ⚒️

| Name | Description | Demo | Popularity |
| ---------- | :---------- | :---------- | :----------: |
| [FaceSwap](https://github.com/deepfakes/faceswap) | Grover is a model for Neural Fake News -- both generation and detection. However, it probably can also be used for other generation tasks. | https://www.youtube.com/watch?v=r1jng79a5xc | [![stars](https://badgen.net/github/stars/deepfakes/faceswap)](https://github.com/deepfakes/faceswap)|
[Face2Face](https://github.com/datitran/face2face-demo) | FaceSwap is a tool that utilizes deep learning to recognize and swap faces in pictures and videos. | N/A | [![stars](https://badgen.net/github/stars/datitran/face2face-demo)](https://github.com/datitran/face2face-demo) |
[Faceswap](https://github.com/MarekKowalski/FaceSwap) | FaceSwap is an app that I have originally created as an exercise for my students in "Mathematics in Multimedia" on the Warsaw University of Technology. | N/A | [![stars](https://badgen.net/github/stars/MarekKowalski/FaceSwap)](https://github.com/MarekKowalski/FaceSwap) |
[Faceswap-GAN](https://github.com/shaoanlu/faceswap-GAN) | Adding Adversarial loss and perceptual loss (VGGface) to deepfakes'(reddit user) auto-encoder architecture. | https://github.com/shaoanlu/faceswap-GAN/blob/master/colab_demo/faceswap-GAN_colab_demo.ipynb | [![stars](https://badgen.net/github/stars/shaoanlu/faceswap-GAN)](https://github.com/shaoanlu/faceswap-GAN) |
[DeepFaceLab](https://github.com/iperov/DeepFaceLab) | DeepFaceLab is a tool that utilizes machine learning to replace faces in videos. | https://www.youtube.com/watch?v=um7q--QEkg4 | [![stars](https://badgen.net/github/stars/iperov/DeepFaceLab)](https://github.com/iperov/DeepFaceLab)|
[Vid2Vid](https://github.com/NVIDIA/vid2vid) | Pytorch implementation for high-resolution (e.g., 2048x1024) photorealistic video-to-video translation.  | https://www.youtube.com/watch?v=5zlcXTCpQqM | [![stars](https://badgen.net/github/stars/NVIDIA/vid2vid)](https://github.com/NVIDIA/vid2vid)|
[DFaker](https://github.com/dfaker/df) | Pytorch implementation for high-resolution (e.g., 2048x1024) photorealistic video-to-video translation.  | N/A | [![stars](https://badgen.net/github/stars/dfaker/df)](https://github.com/dfaker/df)|
[Image Animation](https://github.com/AliaksandrSiarohin/first-order-model) | The videos on the left show the driving videos. The first row on the right for each dataset shows the source videos.  | https://www.youtube.com/watch?v=mUfJOQKdtAk | [![stars](https://badgen.net/github/stars/AliaksandrSiarohin/first-order-model)](https://github.com/AliaksandrSiarohin/first-order-model)|
[Avatarify](https://github.com/alievk/avatarify) | Photorealistic avatars for Skype and Zoom. Democratized. Based on First Order Motion Model..  | https://www.youtube.com/watch?v=lONuXGNqLO0 | [![stars](https://badgen.net/github/stars/alievk/avatarify)](https://github.com/alievk/avatarify)|
[Speech driven animation](https://github.com/DinoMan/speech-driven-animation) | This library implements the end-to-end facial synthesis model.  | N/A | [![stars](https://badgen.net/github/stars/DinoMan/speech-driven-animation)](https://github.com/DinoMan/speech-driven-animation)|


#### 📃 Papers 📃

* [HeadOn: Real-time Reenactment of Human Portrait Videos](https://arxiv.org/pdf/1805.11729.pdf)
* [Face2Face: Real-time Face Capture and Reenactment of RGB Videos](http://gvv.mpi-inf.mpg.de/projects/MZ/Papers/CVPR2016_FF/page.html)
* [Synthesizing Obama: Learning Lip Sync from Audio](https://grail.cs.washington.edu/projects/AudioToObama/siggraph17_obama.pdf)
* [The Creation and Detection of Deepfakes: A Survey](https://arxiv.org/pdf/2004.11138.pdf)

#### 🌐 Webs 🌐

* [DeepFake中文网](https://www.deepfaker.xyz/) :cn:
* [Website for creating deepfake videos with learning](https://deepfakesapp.online/)
* [Deep Fakes Net - Deepfakes Network](https://deep-fakes.net/)
* [Faceswap is the leading free and Open Source multi-platform Deepfakes software](https://faceswap.dev/)
* [Fakening](https://fakening.com/)
* [DeepFakesWeb](https://deepfakesweb.com/)

#### 📺 Videos 📺

* [How to Animate Image with a Video](https://www.youtube.com/watch?v=6W5uoFUIOvk)

## 6. Detection of synthetic content

### 6.1 Detection Text

#### ⚒️ Tools ⚒️

| Name | Description | Demo | Popularity |
| ---------- | :---------- | :---------- | :----------: |
| [Grover](https://github.com/rowanz/grover) | Grover is a model for Neural Fake News -- both generation and detection. However, it probably can also be used for other generation tasks. | [https://grover.allenai.org/](https://grover.allenai.org/) | [![stars](https://badgen.net/github/stars/rowanz/grover)](https://github.com/rowanz/grover)|
[GLTR](https://github.com/HendrikStrobelt/detecting-fake-text) | Detecting text that was generated from large language models (e.g. GPT-2). | http://gltr.io/dist/index.html | [![stars](https://badgen.net/github/stars/HendrikStrobelt/detecting-fake-text)](https://github.com/HendrikStrobelt/detecting-fake-text) |
[fake news detection](https://github.com/nguyenvo09/fake_news_detection_deep_learning) | In this project, we aim to build state-of-the-art deep learning models to detect fake news based on the content of article itself. | [Demo](https://github.com/nguyenvo09/fake_news_detection_deep_learning/blob/master/biGRU_attention.ipynb) | [![stars](https://badgen.net/github/stars/nguyenvo09/fake_news_detection_deep_learning)](https://github.com/nguyenvo09/fake_news_detection_deep_learning) |
[GPTrue or False](https://chrome.google.com/webstore/detail/gptrue-or-false/bikcfchmnacmfhneafnpfekgfhckplfj?hl=en-GB) | Display the likelihood that a sample of text was generated by OpenAI's GPT-2 model. | N/A | N/A |

#### 📃 Papers 📃

* [GLTR: Statistical Detection and Visualization of Generated Text](https://arxiv.org/pdf/1906.04043.pdf)
* [Human and Automatic Detection of Generated Text](https://arxiv.org/pdf/1911.00650.pdf)
* [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/pdf/1909.05858.pdf)
* [The Limitations of Stylometry for Detecting Machine-Generated Fake News](https://arxiv.org/pdf/1908.09805.pdf)

### 6.2 Detection Audio

#### ⚒️ Tools ⚒️

| Name | Description | Demo | Popularity |
| ---------- | :---------- | :---------- | :----------: |
| [Spooded speech detection](https://github.com/elleros/spoofed-speech-detection) | This work is part of the "DDoS Resilient Emergency Dispatch Center" project at the University of Houston, funded by the Department of Homeland Security (DHS). | N/A | [![stars](https://badgen.net/github/stars/elleros/spoofed-speech-detection)](https://github.com/elleros/spoofed-speech-detection)|
[Fake voice detection](https://github.com/dessa-public/fake-voice-detection) | This repository provides the code for a fake audio detection model built using Foundations Atlas. It also includes a pre-trained model and inference code, which you can test on any of your own audio files. | N/A | [![stars](https://badgen.net/github/stars/dessa-public/fake-voice-detection)](https://github.com/dessa-public/fake-voice-detection)
[Fake Voice Detector](https://github.com/kstoneriv3/Fake-Voice-Detection) | For "Deep Learning class" at ETHZ. Evaluate how well the fake voice of Barack Obama 1. confuses the voice verification system, 2. can be detected. | N/A | [![stars](https://badgen.net/github/stars/kstoneriv3/Fake-Voice-Detection)](https://github.com/kstoneriv3/Fake-Voice-Detection)
[CycleGAN Voice Converter](https://leimao.github.io/project/Voice-Converter-CycleGAN/) | An implementation of CycleGAN on human speech conversions | https://leimao.github.io/project/Voice-Converter-CycleGAN/ | [![stars](https://badgen.net/github/stars/leimao/Voice_Converter_CycleGAN)](https://github.com/leimao/Voice_Converter_CycleGAN)


#### 📃 Papers 📃

* [Can We Detect Fake Voice Generated by GANs?](https://github.com/kstoneriv3/Fake-Voice-Detection/blob/master/DLproject_fake_voice_detection.pdf)
* [CycleGAN Voice Converter](https://leimao.github.io/project/Voice-Converter-CycleGAN/)
* [The Rise of Synthetic Audio Deepfakes](https://www.nisos.com/white-papers/rise_synthetic_audio_deepfakes)

### 6.3 Detection Images

#### ⚒️ Tools ⚒️

| Name | Description | Demo | Popularity |
| ---------- | :---------- | :---------- | :----------: |
| [FALdetector](https://github.com/peterwang512/FALdetector) | Detecting Photoshopped Faces by Scripting Photoshop. | https://www.youtube.com/watch?v=TUootD36Xm0 | [![stars](https://badgen.net/github/stars/peterwang512/FALdetector)](https://github.com/peterwang512/FALdetector)|

#### 📃 Papers 📃

* [Detecting Photoshopped Faces by Scripting Photoshop](https://arxiv.org/pdf/1906.05856.pdf)

### 6.4 Detection Videos

#### ⚒️ Tools ⚒️

| Name | Description | Demo | Popularity |
| ---------- | :---------- | :---------- | :----------: |
| [FaceForensics++](https://github.com/ondyari/FaceForensics) | FaceForensics++ is a forensics dataset consisting of 1000 original video sequences that have been manipulated with four automated face manipulation methods: Deepfakes, Face2Face, FaceSwap and NeuralTextures. | https://www.youtube.com/watch?v=x2g48Q2I2ZQ | [![stars](https://badgen.net/github/stars/ondyari/FaceForensics)](https://github.com/ondyari/FaceForensics)|
| [Face Artifacts](https://github.com/danmohaha/CVPRW2019_Face_Artifacts) | Our method is based on the observations that current DeepFake algorithm can only generate images of limited resolutions, which need to be further warped to match the original faces in the source video. | N/A | [![stars](https://badgen.net/github/stars/danmohaha/CVPRW2019_Face_Artifacts)](https://github.com/danmohaha/CVPRW2019_Face_Artifacts)|
[DeepFake-Detection](https://github.com/dessa-public/DeepFake-Detection) | Our Pytorch implementation, conducts extensive experiments to demonstrate that the datasets produced by Google and detailed in the FaceForensics++ paper are not sufficient for making neural networks generalize to detect real-life face manipulation techniques. | http://deepfake-detection.dessa.com/projects | [![stars](https://badgen.net/github/stars/dessa-public/DeepFake-Detection)](https://github.com/dessa-public/DeepFake-Detection)|
[Capsule-Forensics-v2](https://github.com/nii-yamagishilab/Capsule-Forensics-v2) | Implementation of the paper: Use of a Capsule Network to Detect Fake Images and Videos. | N/A | [![stars](https://badgen.net/github/stars/nii-yamagishilab/Capsule-Forensics-v2)](https://github.com/nii-yamagishilab/Capsule-Forensics-v2)|
[ClassNSeg](https://github.com/nii-yamagishilab/ClassNSeg) | Implementation of the paper: Multi-task Learning for Detecting and Segmenting Manipulated Facial Images and Videos (BTAS 2019). | N/A | [![stars](https://badgen.net/github/stars/nii-yamagishilab/ClassNSeg)](https://github.com/nii-yamagishilab/ClassNSeg)|
| [fakeVideoForensics](https://github.com/next-security-lab/fakeVideoForensics) | Fake video detector | https://www.youtube.com/watch?v=8YYRT4lzQgY | [![stars](https://badgen.net/github/stars/next-security-lab/fakeVideoForensics)](https://github.com/next-security-lab/fakeVideoForensics)


#### 📃 Papers 📃

* [Exposing DeepFake Videos By Detecting Face Warping Artifacts](http://www.cs.albany.edu/~lsw/papers/cvprw19a.pdf)
* [DeepFakes: a New Threat to Face Recognition? Assessment and Detection](https://arxiv.org/pdf/1812.08685.pdf)
* [FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/pdf/1901.08971.pdf)
* [Deepfake Video Detection Using Recurrent Neural Networks](https://engineering.purdue.edu/~dgueraco/content/deepfake.pdf)
* [Deep Learning for Deepfakes Creation and Detection: A Survey](https://arxiv.org/pdf/1909.11573v2.pdf)
* [Protecting World Leaders Against Deep Fakes](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Media%20Forensics/Agarwal_Protecting_World_Leaders_Against_Deep_Fakes_CVPRW_2019_paper.pdf)
* [Capsule-Forensics: Using Capsule Networks to Detect Forged Images and Videos](https://arxiv.org/pdf/1810.11215.pdf)
* [DeepFakes and Beyond: A Survey of Face Manipulation and Fake Detection](https://arxiv.org/abs/2001.00179)
* [Media Forensics and DeepFakes:
an overview](https://arxiv.org/pdf/2001.06564.pdf)
* [Everybody’s Talkin’: Let Me Talk as You Want](https://arxiv.org/pdf/2001.05201.pdf)
* [FSGAN: Subject Agnostic Face Swapping and Reenactment](https://arxiv.org/pdf/1908.05932.pdf)
* [Celeb-DF (v2): A New Dataset for DeepFake Forensics](https://arxiv.org/pdf/1909.12962.pdf)
* [Deepfake Video Detection through Optical Flow based CNN](http://openaccess.thecvf.com/content_ICCVW_2019/papers/HBU/Amerini_Deepfake_Video_Detection_through_Optical_Flow_Based_CNN_ICCVW_2019_paper.pdf)
* [MesoNet: a Compact Facial Video Forgery Detection Network](https://arxiv.org/pdf/1809.00888.pdf)
* [Adversarial Deepfakes](https://arxiv.org/pdf/2002.12749.pdf)
* [One-Shot GAN Generated Fake Face Detection](https://arxiv.org/pdf/2003.12244.pdf)
* [Evading Deepfake-Image Detectors with White- and Black-Box Attacks](https://arxiv.org/pdf/2004.00622.pdf)
* [Deepfakes Detection with Automatic Face Weighting](https://arxiv.org/pdf/2004.12027v1.pdf)
* [Unmasking DeepFakes with simple Features](https://arxiv.org/pdf/1911.00686.pdf)
* [VideoForensicsHQ: Detecting High-quality Manipulated Face Videos](https://arxiv.org/pdf/2005.10360.pdf)
* [Disrupting Deepfakes: Adversarial Attacks Against Conditional Image Translation Networks and Facial Manipulation Systems](https://arxiv.org/pdf/2003.01279.pdf)
* [Detecting Deepfake Videos: An Analysis of Three Techniques](https://arxiv.org/pdf/2007.08517v1.pdf)
* [OC-FakeDect: Classifying Deepfakes Using One-class Variational Autoencoder](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w39/Khalid_OC-FakeDect_Classifying_Deepfakes_Using_One-Class_Variational_Autoencoder_CVPRW_2020_paper.pdf)

#### 📺 Videos 📺

* [Deepfake Detection using LSTM and ResNext CNN](https://youtu.be/_q16aJTXVRE)
* [End-To-End AI Video Generation To Bring Fake Humans To Life](https://youtu.be/VxrtbWqwyUk)
* [DeepFake Cyber Security Threats And Opportunities - Matt Lewis](https://www.youtube.com/watch?v=HoYZG1JpZbg)

#### 😎 Awesome 😎

* [Awesome-Deepfakes-Materials](https://github.com/datamllab/awesome-deepfakes-materials)

## 7. Misc

### Articles

* [2020 Guide to Synthetic Media](https://blog.paperspace.com/2020-guide-to-synthetic-media/)
* [Machine Learning Experiments](https://www.linkedin.com/posts/thiago-porto-24004ba8_machinelearning-experiments-deeplearning-ugcPost-6625473356533649408-sl9v/)
* [Building rules in public: Our approach to synthetic & manipulated media](https://blog.twitter.com/en_us/topics/company/2020/new-approach-to-synthetic-and-manipulated-media.html)
* [Contenido Sintético (parte I): generación y detección de audio y texto](https://www.bbvanexttechnologies.com/contenido-sintetico-parte-i-generacion-y-deteccion-de-audio-y-texto/) :es:
* [Contenido Sintético (parte II): generación y detección de imagenes](https://www.bbvanexttechnologies.com/contenido-sintetico-parte-ii-generacion-y-deteccion-de-imagenes/) :es:
* [Contenido Sintético (parte III): generación y detección de vídeo](https://www.bbvanexttechnologies.com/contenido-sintetico-generacion-y-deteccion-de-video/) :es:
* [Fake Candidate](https://edition.cnn.com/2020/02/28/tech/fake-twitter-candidate-2020/index.html)
* [Unraveling the mystery around deepfakes](https://densitydesign.github.io/teaching-dd15/course-results/es03/group08/)
* [Cyber-Security implications of deepfakes](https://newsroom.nccgroup.com/documents/a-report-on-the-cyber-security-implications-of-deepfakes-by-university-college-london-96562)
* [Deepfake Detection Challenge Results: An open initiative to advance AI](https://ai.facebook.com/blog/deepfake-detection-challenge-results-an-open-initiative-to-advance-ai/)
* [The Synthetic Media Landscape](https://www.syntheticmedialandscape.com/)
* [Do (Microtargeted) Deepfakes Have Real Effects on Political Attitudes?](https://journals.sagepub.com/doi/pdf/10.1177/1940161220944364)

### Talks

* [ICML 2019 Synthetic Realities](https://sites.google.com/view/audiovisualfakes-icml2019/)
* [CCN-CERT: Automatizando la detección de contenido Deep Fake](https://www.youtube.com/watch?v=ist4Za3C2DY) :es:
* [TED Talk: Fake videos of real people](https://www.youtube.com/watch?v=o2DDU4g0PRo)
* [Hacking with Skynet](https://www.slideshare.net/GTKlondike/hacking-with-skynet-how-ai-is-empowering-adversaries)
* [RSA: Deep Fakes Are Getting Terrifyingly Real](https://www.youtube.com/watch?v=DGdY-UWOfoo)
* [CVPR 2020 Workshop on media forensics](https://sites.google.com/view/wmediaforensics2020/)

### Challenges

* [NIST: Media Forensics Challenge 2019](https://www.nist.gov/itl/iad/mig/media-forensics-challenge-2019-0)
* [ASVspoof: Automatic Speaker Verification](https://www.asvspoof.org/)
* [Kaggle: DeepFake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge)
* [Fake News Challenge](http://www.fakenewschallenge.org/)
* [Xente: Fraud detection challenge](https://zindi.africa/competitions/xente-fraud-detection-challenge)
* [Chalearn Multi-modal Cross-ethnicity Face anti-spoofing Recognition Challenge](https://competitions.codalab.org/competitions/22036)

### Forums

* [Reddit: MediaSynthesis](https://www.reddit.com/r/MediaSynthesis/)
* [Reddit: Digital Manipulation](https://www.reddit.com/r/Digital_Manipulation/)
* [MrDeepFake Forums](https://mrdeepfakes.com/forums/) 🔞
* [AIVillage](aivillage.slack.com)
