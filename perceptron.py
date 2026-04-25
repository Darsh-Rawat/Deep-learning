class Perceptron : 
  def __init__(self, eta=0.1) : 
    self.eta = eta
    self.weights = None
    self.bias = None

  def fit(self, X, y, epochs=10) : 
    # Generate Random weights and bias based on shape
    self.weights = np.random.random(size=X.shape[1])
    self.bias = np.random.random()

    # Training
    for i in range(epochs) : 
      pred = -1
      for idx,row in enumerate(X) : 
        z = (row[0] * self.weights[0]) + (row[1] * self.weights[1]) + self.bias

        # Step Function (heavside)
        if z >= 0 : 
          pred = 1
        else : 
          pred = 0
        
        if y[idx] != pred : 
          # Update Weights
          self.weights[0] = self.weights[0] + (0.1 * row[0] * (y[idx] - pred))
          self.weights[1] = self.weights[1] + (0.1 * row[1] * (y[idx] - pred))

          # Update Bias
          self.bias = self.bias + (0.1 * (y[idx] - pred))


  def predict(self, x) : 
    pred = []

    if np.ndim(x) == 1 : 
      z = (x[0] * self.weights[0]) + (x[1] * self.weights[1]) + self.bias
      
      if z >= 0 : 
        pred.append(1)
      else : 
        pred.append(0)

    elif np.ndim(x) == 2 : 
      for row in x : 
        z = (row[0] * self.weights[0]) + (row[1] * self.weights[1]) + self.bias

        if z >= 0 : 
          pred.append(1)
        else : 
          pred.append(0)
        
    return pred
