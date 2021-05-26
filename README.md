# Letter CGAN

### Structure:
* main.py - startup file
* nets.py - contains all networks
* tools.py - utils (core functions, classes, etc.)
* paint.py - interface (small tkinter application)
* decoder - letter conversion module

### Short description of the algorithm:
1. Get input tensor (paint.py)
2. Apply ensemble and get class label (tools.py - Ensemble)
3. Decoder label conversion (decoder.py)
3. Create multiple samples per label and choose the best one (tools.py - GenerativeAlgorithm)
4. Show result

#### CGAN samples
![samples](https://github.com/gitDev-ms/letter-cgan/blob/main/images/samples.png)

### Application keys binding:
* F5 - reset canvas
* Enter - apply algorithm

#### Overall results
![samples](https://github.com/gitDev-ms/letter-cgan/blob/main/images/transformed-samples.png)
