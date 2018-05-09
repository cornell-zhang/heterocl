class Type():
  def __init__(self, bits = None, frac = None):
    self.bits = bits
    self.frac = frac

  @property
  def bits(self):
    return self.bits

  @property
  def frac(self):
    return self.frac

class Int(Type):
  pass

class UInt(Type):
  pass

class Float(Type):
  pass

class Double(Type):
  pass

class Fixed(Type):
  pass
