# Bitcoin Node + electrs Infrastructure

Local Bitcoin Core node with Blockstream electrs (Esplora-compatible REST API) for the AML investigation platform.

## Quick Start

```bash
cd infra

# Copy and configure
cp .env.example .env
# Edit .env to choose network (signet or testnet4)

# Start services
docker-compose up -d

# Monitor sync progress
docker-compose logs -f bitcoind
docker-compose logs -f electrs
```

## Network Options

### Signet (Recommended for Development)
- **Sync time**: 5-15 minutes
- **Disk**: ~2GB
- **Ports**: RPC 38332, P2P 38333
- **Use case**: Fast iteration, stable testnet

```bash
BTC_NETWORK=signet
RPC_PORT=38332
P2P_PORT=38333
```

### Testnet4
- **Sync time**: 1-3 hours
- **Disk**: ~30GB
- **Ports**: RPC 48332, P2P 48333
- **Use case**: More realistic tx volume

```bash
BTC_NETWORK=testnet4
RPC_PORT=48332
P2P_PORT=48333
```

## Monitoring Sync Status

### Bitcoin Core
```bash
# Check sync progress
docker exec aml-bitcoind bitcoin-cli -rpcuser=aml -rpcpassword=amlpass -rpcport=${RPC_PORT} getblockchaininfo

# Watch blocks
docker exec aml-bitcoind bitcoin-cli -rpcuser=aml -rpcpassword=amlpass -rpcport=${RPC_PORT} getblockcount
```

### electrs (Esplora API)
```bash
# Check tip height (should match bitcoind once synced)
curl http://localhost:3000/blocks/tip/height

# Health check
curl http://localhost:3000/blocks/tip/hash

# Get address info (example signet address)
curl http://localhost:3000/address/tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx
```

## API Endpoints (electrs)

All endpoints are prefix-less (no `/api`):

- `GET /blocks/tip/height` — current block height
- `GET /blocks/tip/hash` — current block hash
- `GET /tx/:txid` — transaction details
- `GET /tx/:txid/outspends` — outputs spent status (for forward tracing)
- `GET /address/:address` — address summary (chain_stats, mempool_stats)
- `GET /address/:address/txs` — address transactions (paginated, 50 mempool + 25 confirmed per page)
- `GET /address/:address/txs/chain/:last_seen_txid` — pagination continuation

## Configuration Notes

**RPC Authentication**: Uses `rpcuser`/`rpcpassword` (suitable for local development). RPC is restricted to the compose network subnet `172.30.0.0/16`.

**electrs Cookie Auth**: Configured with `--cookie=aml:amlpass` to match bitcoind credentials.

**Network Isolation**: Services communicate over the `chain-network` bridge. Backend can reach electrs at:
- If backend runs on host: `http://localhost:3000`
- If backend runs in Docker: `http://aml-electrs:3000` (requires shared network)

## Disk Space

- **Signet**: ~2GB (blocks) + ~500MB (electrs index)
- **Testnet4**: ~30GB (blocks) + ~3GB (electrs index)

Volumes are persistent across restarts: `bitcoin_data`, `electrs_data`.

## Health Checks

```bash
# Check all services
docker-compose ps

# electrs healthcheck (built-in)
# Status: healthy once synced
docker inspect aml-electrs | grep -A 10 Health

# Full restart if needed
docker-compose down
docker-compose up -d
```

## Troubleshooting

**electrs can't connect to bitcoind**:
- Check bitcoind is fully started: `docker-compose logs bitcoind | grep "init message: Done loading"`
- Verify RPC works: `docker exec aml-bitcoind bitcoin-cli -rpcuser=aml -rpcpassword=amlpass getblockchaininfo`

**electrs sync stalled**:
- electrs indexes after bitcoind sync completes
- Check progress: `docker-compose logs electrs | grep "indexed"`
- Typical signet indexing: 2-5 minutes after block sync

**Out of disk space**:
- Check usage: `docker system df -v`
- Prune old data: `docker volume prune` (WARNING: destroys volumes)

## Security Notes

⚠️ **Local development only**: 
- RPC credentials are hardcoded for convenience
- `rpcallowip` is restricted to the compose network
- Do not expose RPC or P2P ports to the internet
- Do not use this setup for mainnet with real funds

## Integration with Backend

Set in backend `.env`:

```bash
# If backend runs on host
ESPLORA_URL=http://localhost:3000

# If backend runs in Docker (add to main docker-compose.yml networks)
ESPLORA_URL=http://aml-electrs:3000
```

See main `README.md` for full backend setup.
