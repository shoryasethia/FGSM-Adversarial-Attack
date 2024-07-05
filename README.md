# FGSM-Adversarial-Attack

## FGSM : Fast Gradient Sign Method
It is an efficient technique used in adversarial machine learning to create adversarial examples, which are inputs to machine learning models that are intentionally designed to cause the model to make a mistake. FGSM is particularly notable for its simplicity and effectiveness in generating these adversarial examples.

## How FGSM Works ?
FGSM perturbs the input data in the direction of the gradient of the loss function with respect to the input. The core idea is to make minimal changes to the input that maximize the loss, effectively fooling the model. The perturbation is controlled by a parameter, epsilon (ε), which determines the magnitude of the perturbation.

## Steps to Implement FGSM
* **Compute the Gradient:** Calculate the gradient of the loss function with respect to the input data.
* **Determine Perturbation:** Use the sign of the gradients to determine the direction in which to perturb the input.
* **Apply Perturbation:** Add the perturbation to the input data, scaled by epsilon (ε)

## Mathematical Equation
Moreover, we try to maximize the loss wrt image pixels.
![Equation](https://github.com/shoryasethia/FGSM-Adversarial-Attack/blob/main/FGSM-Equation.png)
## FGSM Tensorflow Implementation
```
import tensorflow as tf

def fgsm_attack(model, x, y, epsilon):
    with tf.GradientTape() as tape:
        tape.watch(x)
        prediction = model(x)
        loss = tf.keras.losses.categorical_crossentropy(y, prediction)
    gradient = tape.gradient(loss, x)
    perturbation = epsilon * tf.sign(gradient)
    x_adv = x + perturbation
    x_adv = tf.clip_by_value(x_adv, 0, 1)  # Ensure the values are in the valid range
    return x_adv
```
## Paper
[Explaining and Harnessing Adversarial Examples](https://github.com/shoryasethia/FGSM-Adversarial-Attack/blob/main/1412.6572v3.pdf) by Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy.

> **Check out [this](https://github.com/shoryasethia/Adversarial-Attack-Defence) repo, where I used Denoising AutoEncoder + Block Switch model to defend against an FGSM adversarial attack on CIFAR10. To an extent DAE+BS was able to defend and gave more correct predictions.**
> 
### Author : [@shoryasethia](https://github.com/shoryasethia)
> If you liked anything, do give this repo a star.
