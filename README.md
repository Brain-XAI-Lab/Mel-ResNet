<h1>Mel-ResNet</h1>
ResNet for classifying Mel-spectrograms

<img width="1000" alt="Mel-ResNet figure" src="https://github.com/Brain-XAI-Lab/Mel-ResNet/assets/94499717/e2e3a490-7a09-4014-8d44-653de56840b9">
<ul>
    <li>We developed a generative model that inputs EEG data during Inner Speech (Imagined Speech) and the corresponding mel spectrogram of Spoken Speech (target) into a GAN.</li>
    <li>Through this approach, the mel spectrogram generated from the EEG is fed into a ResNet trained on actual speech mel spectrograms to predict the imagined word.</li>
</ul>
<br>

<h2>Requirements</h2>
`Python >= 3.7`<br>

All the codes are written in Python 3.7.

You can install the libraries used in our project by running the following command:
```bash
pip install -r requirements.txt
```
<br>

<h2>Dataset</h2>
<p>We extracted word utterance recordings for a total of <strong>13 classes</strong> using the voices of 5 contributors and TTS technology.<br>
Additionally, to address any data scarcity issues, we applied augmentation techniques such as time stretching, pitch shifting, and adding noise.</p>
The pairs of recorded words are as follows:
<br><br>
<ul>
    <li>Call</li>
    <li>Camera</li>
    <li>Down</li>
    <li>Left</li>
    <li>Message</li>
    <li>Music</li>
    <li>Off</li>
    <li>On</li>
    <li>Receive</li>
    <li>Right</li>
    <li>Turn</li>
    <li>Up</li>
    <li>Volume</li>
</ul>
<br>

<h2>Model & Training (ongoing)</h2>
<p><strong>ResNet-50</strong></p>
<img src="https://github.com/user-attachments/assets/d7e55c25-702e-48f5-8ecb-00b02cf92b85">
<p>(We are continuously collecting data and will need to undergo a hyperparameter tuning process after training.)</p>
<br>

<h2>Results (ongoing)</h2>
<p>Performance metrics : <strong>Accuracy, F1 score</strong></p>
<p>(learning curve and metrics will be here)</p>
<br>

<h2>References</h2>
