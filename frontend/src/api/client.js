const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class APIClient {
  async get(endpoint) {
    const response = await fetch(`${API_BASE_URL}${endpoint}`);
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const message = errorData.detail || response.statusText;
      throw new Error(message);
    }
    return response.json();
  }

  async post(endpoint, data) {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }
    return response.json();
  }

  async delete(endpoint) {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }
    return response.json();
  }

  // Elliptic++ endpoints
  async getEppExamples() {
    return this.get('/api/epp/examples');
  }

  async getEppWallet(address) {
    return this.get(`/api/epp/wallet/${address}`);
  }

  async getEppAddress(address) {
    return this.get(`/api/epp/address/${address}`);
  }

  async getEppTx(txid) {
    return this.get(`/api/epp/tx/${txid}`);
  }

  async getEppTxEdges(txid, focalAddress, max = 10) {
    return this.get(`/api/epp/tx/${txid}/edges?focal=${focalAddress}&max=${max}`);
  }

  async getEppWalletExplainGnn(address) {
    return this.get(`/api/epp/wallet/${address}/explain/gnn`);
  }

  // Chain endpoints
  async getChainStatus() {
    return this.get('/api/chain/status');
  }

  async getChainSubgraph(address, k = 1, maxTxs = null) {
    const params = maxTxs ? `?k=${k}&max_txs=${maxTxs}` : `?k=${k}`;
    return this.get(`/api/chain/subgraph/${address}${params}`);
  }

  async getChainWallet(address) {
    return this.get(`/api/chain/wallet/${address}`);
  }

  async getChainAddress(address) {
    return this.get(`/api/chain/address/${address}`);
  }

  async getChainAddressTxsPage(address, cursor = null, pageSize = 25) {
    const params = cursor ? `?cursor=${cursor}&page_size=${pageSize}` : `?page_size=${pageSize}`;
    return this.get(`/api/chain/address/${address}/transactions${params}`);
  }

  async getChainTx(txid) {
    return this.get(`/api/chain/tx/${txid}`);
  }

  async getChainTxEdges(txid, focalAddress, max = 10) {
    return this.get(`/api/chain/tx/${txid}/edges?focal=${focalAddress}&max=${max}`);
  }

  async getChainAddressMixing(address) {
    return this.get(`/api/chain/address/${address}/mixing`);
  }

  async getChainWalletLiveScore(address) {
    return this.get(`/api/chain/wallet/${address}/live-score`);
  }

  // Investigation endpoints
  async listInvestigations(analystId = 'default') {
    return this.get(`/api/investigations?analyst_id=${analystId}`);
  }

  async getInvestigation(id) {
    return this.get(`/api/investigations/${id}`);
  }

  async saveInvestigation(data) {
    return this.post('/api/investigations', data);
  }

  async deleteInvestigation(id) {
    return this.delete(`/api/investigations/${id}`);
  }

  // Model
  async getModelInfo() {
    return this.get('/api/model/info');
  }

  // Health check
  async healthCheck() {
    return this.get('/health');
  }
}

export default new APIClient();
