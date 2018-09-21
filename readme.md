# Edge Enhancement GAN

This is the subsidiary network of the project "Everybody Dance Now". We adopt the GlobalGenerator in pix2pixHD project with a slight twist of resblock numbers to cope with smaller input face regions. Here's the result of NTU-256 video frames after training with 20000 batches (batch size 64)

![enhanced_full](samples/20000_enhanced_full.png)
![enhanced_head](samples/20000_enhanced_head.png)

The quality of our data is bad, so the result looks weird with random colored noise. Nontheless, the fact that my code runs smoothly on Windows PC is satisfying enough (for now).
