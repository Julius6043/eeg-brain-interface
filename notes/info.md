## Info
* 3 Class Problem (0-Back is underrepresented; unbalanced)

* Split the Data in Windows (Epochs). One Window is between 2 and 10 Seconds and with 50% overlab.
* For Deeplearning the 3D format Data could be used.
* For ML each Window is calculated in a Feature Vector and the 2D Format is used.
* There is also a label vector with the label (N-Back[1,2,3]) with a element for each window.

* Use the welch Method in scipy for the PSD
* sklearn for Feature selection
* pick k-best