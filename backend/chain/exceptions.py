"""Exception hierarchy for chain data operations"""


class ChainError(Exception):
    """Base exception for chain operations"""
    pass


class InvalidAddress(ChainError):
    """Address format is invalid or network prefix doesn't match configured network"""
    pass


class NotFound(ChainError):
    """Transaction or address not found"""
    pass


class ChainUnavailable(ChainError):
    """Chain data source (electrs/bitcoind) is unreachable or timed out"""
    pass
