import logging
def get_custom_logger(name, level=logging.INFO):
  """ Example of a custom logger.
  
    This function takes in two parameters: name and level and logs to console.
    The place to log in this case is defined by the handler which we set
    to logging.StreamHandler().
    
    Args:
      name: Name for the logger.
      level: Minimum level for messages to be logged
  """
  logger = logging.getLogger(name)
  logger.setLevel(level)

  if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("Optimizely({}): %(levelname)s - %(asctime)s - %(message)s ".format(name))
    handler.setFormatter(formatter)
    logger.addHandler(handler)

  return logger


logger = get_custom_logger(name="digital_twins")