# micrograd_plus

Inspired by Andrej Karpathy's micrograd, micrograd_plus is an autograd engine over scalar-valued Directed Acyclic Graphs. It extends micrograd by implementing additional activation functions, like sigmoid and relu, in addition to the included tanh along with a draw function that draws a DAG with all neurons and their gradients. micrograd_plus can be used to make neural networks for binary classification tasks. <br>

Examples of DAGs with different activation functions <br>

```ruby
# tanh
# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'
o.backward()
o.draw()
```

<img width="1229" alt="Screenshot 2024-08-02 at 4 02 23 AM" src="https://github.com/user-attachments/assets/412097e9-92c7-4060-b27e-4841918ac470"> <br>

```ruby
# sigmoid
# inputs x1,x2
x1 = Value(3.0, label='x1')
x2 = Value(2.0, label='x2')
# weights w1,w2
w1 = Value(5.0, label='w1')
w2 = Value(-4.0, label='w2')
# bias of the neuron
b = Value(-2, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'
o.backward()
o.draw()
```

<img width="1228" alt="Screenshot 2024-08-02 at 4 05 30 AM" src="https://github.com/user-attachments/assets/01f35d55-9e4f-48f6-8cc0-b51d08807412"> <br>

Here's an example of a nerual net being trained to classify points in a very simple syntetically generated dataset<br>
![Screenshot 2024-08-02 at 5 04 23 PM](https://github.com/user-attachments/assets/fcb4a0ab-a6a6-4bb3-8252-d67bf03de0db)
![Screenshot 2024-08-02 at 5 05 08 PM](https://github.com/user-attachments/assets/97578efd-4389-45d3-bc3b-bdda829ac399)

Here's an example of a nerual net being trained on the real world iris-flower dataset to classify flower species based on sepal length and width<br>
![Screenshot 2024-08-02 at 5 17 58 PM](https://github.com/user-attachments/assets/ef997a4b-de18-4f11-9169-bc32fc92ddf5)
![Screenshot 2024-08-02 at 5 18 41 PM](https://github.com/user-attachments/assets/3c6d6113-f83d-4aee-b169-d92de0b2b894)
