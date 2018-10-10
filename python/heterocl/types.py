class Type():
  def __init__(self, bits = 32, fracs = 0):
    self.bits = bits
    self.fracs = fracs

  @property
  def bits(self):
    return self.bits

  @property
  def fracs(self):
    return self.fracs

class Int(Type):
  def __repr__(self):
    return "Int(" + str(self.bits) + ")"

class UInt(Type):
  def __repr__(self):
    return "UInt(" + str(self.bits) + ")"

class Float(Type):
  def __repr__(self):
    return "Float()"

class Double(Type):
  def __repr__(self):
    return "Double()"

class Fixed(Type):
  def __repr__(self):
    return "Fixed(" + str(self.bits) + ", " + str(self.fracs) + ")"

class UFixed(Type):
  def __repr__(self):
    return "UFixed(" + str(self.bits) + ", " + str(self.fracs) + ")"
