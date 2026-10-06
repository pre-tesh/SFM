Hello , this project is a part of a bigger project , which is  generating  dense 3d point clouds of objects .
The problem of 3d reconstruction has in he past years developed many fairly advanced and ML , DL based approaches , but i like to learn and disect the thoughts behind teh thinking  that researchers and those godly engineers have put into thinking and making everything happen . 
so this is my attempt to implement the SFM pipeline in python. 
Because I believe that will help me learn a lot .

The final goal that i have will be generating a dense 3d point clouds of 
1.a bulbasaur plushie  
3.a kurapika figurine 

SFM stands for structure from motion .
SO we  are trying to generate a 3d structure from multiple 2d images .

***STEPS***

1. USED SIFT , i dont plan on writing SIFT from scratch here , because it is fairly complex with all the maths that scared me when i tried to read LOWE's SIFT paper , so we resort to using
OPENCV'S SIFT implementation. 
Lets take a moment to clap and appreciate OPEN SOURCE . 'Clap clap".

2. Next we find the matching points between differnt images , that is say 
image 1 has n features 
image 2 has m features 
then we find the distance of each feature in img1 with img2 and then find the closest match,
and store them in a list .

3. Next we use  the 8 point algorithm and ransac , to eliminate fake matches , and also 
get the fundamental matrix.
so we pick random 8 points to calculate the Fundamental matrix F , then verify and select the best F , i.e the F whch gets us the most amount of inliers , we also then store the inliers .

4. Camera pose estimation :


