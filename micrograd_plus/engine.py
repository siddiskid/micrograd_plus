import math
import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph

class Value:
  #represents a value object - a scalar value and it's gradient
  def __init__(self, data, _children=(), _op="", label = ""):
    self.data = data
    self.grad = 0.0
    self._op = _op
    self.label = label
    self._prev = set(_children)
    self._backward = lambda: None

  # backward pass (calculate grads)
  def backward(self):
    # topological sort
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad = 1
    for node in reversed(topo):
      node._backward()

  # activation functions
  def sigmoid(self):
    x = self.data
    t = 1 / (1 + math.exp(-x))
    out = Value(t, (self,), "sigmoid")

    def _backward():
      self.grad += t * (1 - t) * out.grad
    out._backward = _backward

    return out

  def relu(self):
    t = 0 if self.data < 0 else self.data
    out = Value(t, (self,), "relu")

    def _backward():
      self.grad += (t > 0) * out.grad
    out._backward = _backward

    return out
  
  def tanh(self):
    x = self.data
    t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
    out = Value(t, (self,), "tanh")

    def _backward():
      self.grad += (1 - t ** 2) * out.grad
    out._backward = _backward
    
    return out 

  # draw function to visualize
  def draw(self):
    def trace(root):
      # builds a set of all nodes and edges in a graph
      nodes, edges = set(), set()
      def build(v):
        if v not in nodes:
          nodes.add(v)
          for child in v._prev:
            edges.add((child, v))
            build(child)
      build(root)
      return nodes, edges
    
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right

    nodes, edges = trace(self)
    for n in nodes:
      uid = str(id(n))
      # for any value in the graph, create a rectangular ('record') node for it
      dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
      if n._op:
        # if this value is a result of some operation, create an op node for it
        dot.node(name = uid + n._op, label = n._op)
        # and connect this node to it
        dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
      # connect n1 to the op node of n2
      dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
  
  # math
  def exp(self):
    x = self.data
    e = math.exp(x)
    out = Value(e, (self,), "exp")

    def _backward():
      self.grad = e * out.grad
    out._backward = _backward

    return out
  
  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), "+")

    def _backward():
      self.grad += 1.0 * out.grad # previously = 1.0 * out.grad which caused backpropogation bug when same node used twice
      other.grad += 1.0 * out.grad
    out._backward = _backward

    return out
  
  def __radd__(self, other):
    return self + other

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), "*")

    def _backward():
      self.grad += other.data * out.grad 
      other.grad += self.data * out.grad
    out._backward = _backward

    return out
  
  def __rmul__(self, other):
    return self * other
  
  def __neg__(self):
    return self * -1
  
  def __sub__(self, other):
    return self + (-other)
  
  def __truediv__(self, other):
    return self * other**-1
  
  def __pow__(self, other):
    out = Value(self.data**other, (self, ), f"**{other}")

    def _backward():
      self.grad += (other * (self.data ** (other - 1))) * out.grad
    out._backward = _backward

    return out
  
  def __repr__(self):
    return f"Value(data={self.data})"

