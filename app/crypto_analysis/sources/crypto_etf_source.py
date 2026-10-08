"""通过MCP获取加密资产现货ETF资金流，不直接调用REST接口。"""

from pathlib import Path
from typing import Any

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.mcp_client.manager import MCPClientManager


class CryptoETFSettings(BaseSettings):
    """MCP地址由实际服务提供方确定，不从REST地址猜测。"""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[3] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    crypto_etf_mcp_api_key: SecretStr = SecretStr("")
    crypto_etf_mcp_url: str = "https://mcp.cryptoetf.today/api/mcp"


async def get_crypto_etf_flows(target_asset: str) -> dict[str, Any]:
    """通过MCP的get_asset_flows取得原始ETF流量，交给计算器处理。"""
    asset = target_asset.lower()
    supported = {
        "btc",
        "eth",
        "sol",
        "xrp",
        "hyp",
        "doge",
        "link",
        "avax",
        "hbar",
        "ltc",
        "bnb",
        "dot",
        "sui",
    }
    if asset not in supported:
        raise ValueError(
            "目标资产不在ETF服务支持列表中，不能据此认定该资产不存在现货ETF"
        )
    config = CryptoETFSettings()
    endpoint = config.crypto_etf_mcp_url
    api_key = config.crypto_etf_mcp_api_key.get_secret_value()
    if not api_key:
        raise RuntimeError("未配置 CRYPTO_ETF_MCP_API_KEY")
    if not endpoint:
        raise RuntimeError("未配置 CRYPTO_ETF_MCP_URL")
    async with MCPClientManager(
        {"crypto_etf": endpoint}, {"crypto_etf": {"Authorization": f"Bearer {api_key}"}}
    ) as client:
        payload = await client.call_json_tool(
            "crypto_etf", "get_asset_flows", {"asset": asset}
        )
    return {
        "payload": payload,
        "source": f"https://api.cryptoetf.today/api/v1/flows/{asset}",
    }
