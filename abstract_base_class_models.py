from abc import ABC, abstractmethod, abstractproperty
import numpy as np

class CompartmentModel(ABC):
	@abstractmethod
	def ab(self):
		pass

	@abstractmethod
	def num_parameters(n_params):
		pass

	@abstractproperty
	def states(self):
		pass

	@abstractproperty
	def parameters(self):
		pass

	@abstractproperty
	def inputs(self):
		pass

class OneCompartmentModel(CompartmentModel):
	def __init__(self, states, parameters, inputs, n_params):
		self.states = states
		self.parameters = parameters
		self.inputs = inputs
		self.n_params = n_params

	def ab(self, i):
		gi, b = self.states[i]
		ka, k = self.parameters
		dGI_dt = self.inputs[i] - 1 * ka * gi
		dC_dt = ka * gi - k * b
		return np.column_stack((dGI_dt.flatten(), dC_dt.flatten()))

	def num_parameters(n_params):
		return self.n_params

	def states(self):
		return self.states

	def parameters(self):
		return self.parameters

	def inputs(self):
		return self.inputs

class TwoCompartmentModel(CompartmentModel):
	def __init__(self, states, parameters, inputs, n_params):
		self.states = states
		self.parameters = parameters
		self.inputs = inputs
		self.n_params = n_params

	def ab(self, i):
		gi, c, p = self.states[i]
		ka, k10, k12, k21 = self.parameters
		dGI_dt = self.inputs[i] - 1 * ka * gi
		dC_dt = ka * gi + k21 * p - (k10 + k12) * c
		dP_dt = k12 * c - k21 * p
		return np.column_stack((dGI_dt.flatten(), dC_dt.flatten(), dP_dt.flatten()))

	def num_parameters(n_params):
		return self.n_params

	def states(self):
		return self.states

	def parameters(self):
		return self.parameters

	def inputs(self):
		return self.inputs

class ThreeCompartmentModel(CompartmentModel):
	def __init__(self, states, parameters, inputs, n_params):
		self.states = states
		self.parameters = parameters
		self.inputs = inputs
		self.n_params = n_params

	def ab(self, i):
		gi, c, p, dt = self.states[i]
		ka, k10, k12, k21, k13, k31 = self.parameters
		dGI_dt = self.inputs[i] - 1 * ka * gi
		dC_dt = ka * gi + k21 * p + k31 * dt - k10 * c - k12 * c - k13 * dt
		dP_dt = k12 * c - k21 * p
		dDT_dt = k13 * c - k31 * dt
		return np.column_stack((dGI_dt.flatten(), dC_dt.flatten(), dP_dt.flatten(), dDT_dt.flatten()))

	def num_parameters(n_params):
		return self.n_params

	def states(self):
		return self.states

	def parameters(self):
		return self.parameters

	def inputs(self):
		return self.inputs