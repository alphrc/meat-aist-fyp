from __future__ import annotations

import numpy as np
import torch

from typing import Literal
from abc import abstractmethod


class Shape(tuple):
    def __new__(cls, *args):
        return super().__new__(cls, *args)


class Port:
    def __init__(self, node: Node, shape: Shape, direction: Literal["in", "out"]):
        self.node = node
        self.shape = shape
        self.direction = direction
        if direction == "in":
            self.in_buffers: dict[Channel, np.ndarray] = {}
            self.in_weights: dict[Channel, float] = {}
        elif direction == "out":
            self.out_channels: list[Channel] = []
    
    def add_in_channel(self, channel: Channel, weight: float):
        assert self.direction == "in", "Cannot add in channel to out port"
        assert channel.shape == self.shape, "Channel shape must match port shape"
        assert channel not in self.in_buffers, "Channel already added"
        self.in_buffers[channel] = None
        self.in_weights[channel] = weight

    def add_out_channel(self, channel: Channel):
        self.out_channels.append(channel)

    def receive(self, channel: Channel):
        self.in_buffers[channel] = channel.buffer
        if all(in_buffer is not None for in_buffer in self.in_buffers.values()):
            self.in_buffer = sum(in_buffer * in_weight for in_buffer, in_weight in zip(self.in_buffers.values(), self.in_weights.values()))
            self.node.receive(self)
            self.in_buffers = {channel: None for channel in self.in_buffers}
            self.in_buffer = None

    def send(self, data: np.ndarray):
        self.out_buffer = data
        for out_channel in self.out_channels:
            out_channel.receive()
        self.out_buffer = None


class Node:
    def __init__(self, input_shapes: list[Shape], output_shapes: list[Shape]):
        self.in_ports = [Port(self, shape, "in") for shape in input_shapes]
        self.out_ports = [Port(self, shape, "out") for shape in output_shapes]

        self.in_buffers: dict[Port, np.ndarray] = {port: None for port in self.in_ports}
        self.output_buffers: dict[Port, np.ndarray] = {port: None for port in self.out_ports}

    def receive(self, in_port: Port):
        self.in_buffers[in_port] = in_port.in_buffer
        if all(buffer is not None for buffer in self.in_buffers.values()):
            self.forward()
            self.in_buffers = {port: None for port in self.in_buffers}

    def send(self):
        for out_port, out_buffer in self.output_buffers.items():
            out_port.send(out_buffer)
        self.output_buffers = {port: None for port in self.output_buffers}

    @abstractmethod
    def forward(self): # calculate and set output buffers
        pass


class Channel:
    def __init__(self, shape: Shape, in_port: Port, out_port: Port):
        self.shape = shape
        self.in_port = in_port
        self.out_port = out_port

    def receive(self):
        self.buffer = self.out_port.out_buffer
        self.in_port.receive(self)
        self.buffer = None


class Module(Node):
    def __init__(self, input_shapes: list[Shape], output_shapes: list[Shape]):
        super().__init__(input_shapes, output_shapes)

    def forward(self): # [TODO]
        pass


class Hypergraph(Node):
    def __init__(self, input_shapes: list[Shape], output_shapes: list[Shape], nodes: list[Node], channels: list[Channel]):
        super().__init__(input_shapes, output_shapes)
        self.nodes = nodes
        self.channels = channels


class Model(Hypergraph):
    pass



if __name__ == "__main__":
    pass