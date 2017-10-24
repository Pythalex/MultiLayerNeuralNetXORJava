# Multilayer Neural Net XOR

![alt](http://sj.uploads.im/t/hLmPA.png)
___

### Why ?

I had a hard time to make it work, as this is a simple problem used to illustrate the use of multilayer neural networks, often, solutions are presented but not the code behind it. And as a non-user of machine learning module for python, I couldn't understand many of the forum's q&a about XOR problem because of this.

So, because I think other people might be in the same situation, I put my implementation on github. I'd be really glad if it could help somebody out there, we never know.

___

### The code

The program is made to be generalized to x hidden neurons in only one hidden layer and y output neurons. The reason is the algorithm sometimes returned bad results with only 2 hidden neurons (as in many XOR solutions provided by many neural net courses), and as the reason turned out to be a consequence of having only 2 hidden neurons, I reorganized the code to set a specified number of internal neurons. Thus, the main function repeats a certain number of times (1000 by default) the learning process to calculate the missing rate of the learning algorithm. As it can be observed, we have these results for n hidden neurons between 2 and 5 :

2 hidden neurons : 0.297% error rate

3 hidden neurons : 0.064% error rate

4 hidden neurons : 0.003% error rate

5 hidden neurons : 0.0%   error rate

(Of course, the last error rate isn't really correct because of the precision, but you surely got the point)
