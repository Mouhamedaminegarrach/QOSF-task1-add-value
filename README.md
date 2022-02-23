# QOSF-task1-add-value

Task 1 Add values

For this problem, you want to find a positive integer that can be composed of the summation of a subset of the input vector, for example:
[1,3,6,4,2],   and we need to find the number 6

the possible solutions are:
- [1,3,2],
- [4,2],
- [6]
For this challenge, consider as input a vector of positive integers and a positive integer, generate a quantum circuit that indicates with higher probability the subset(s) that manage to obtain the number with their sum.
Tip consider a QRAM to save the input vector and the encoding basis, using the before example that could be, we need n qubits for the length of the vector and m qubits for the length of the bits. Consider that we will use n = 5 qubits for the address, so the state |10000⟩ represents the index 1, the state |01000⟩  represents the index 2 and so on. And m = 3, because the number we need is 6 and in binary is 110, so we can use the bases encoding and  the state  result is |110⟩
In view of the above, the following should be carried out
|index of the vector⟩|value of the index⟩
Consider this format, we  need 5 qubits for the index and 3 qubits of the values, i.e:
1 = > |10000001>
3 = > |01000011>
6 = > |00100110>
4 = > |00010100>
2 = > |00001010>
And you want to find 6. The green part is the index state, and the red is the value.
For this example, you need to find an oracle where is the state six in the  red qubits is a correct answer.
The output could be 
- [1,3,2]  = > |11001> ,
- [4,2]  = > |00011>  ,
- [6]  = > |00100> 
Hint: For this task you can make use of the Adder by Draper, a general quantum circuit to make this proposal can be found  here, but instead of adding 2 numbers find a way to accommodate the request.

Based on the diagram, the QFT and QFT-1 are used and U1 =  .  Then  the values  to be the binary number  depends on the U1 gates. Examples of this you can see in the images, consider the green cat  as |0> and purple cat as |1>. 

The challenge:

Design a quantum circuit that finds the subsets where the sum is equal to the value 16 in the following vector [5,7,8,9,1]
References 
Steven A. Cuccaro, Thomas G. Draper, Samuel A. Kutin, David Petrie Moulton (2004). A new quantum ripple-carry addition circuit. (https://arxiv.org/abs/quant-ph/0410184)
Thomas G. Draper (2000). Addition on a Quantum Computer (https://arxiv.org/abs/quant-ph/0008033)
