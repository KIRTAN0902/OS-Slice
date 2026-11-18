"""
OS-Slice Data Generation & Operational Boundary Masking Module
Generates high-density enterprise structured JSON extraction calibration and evaluation datasets.
Computes token-level operational boundary masks separating structural syntax from semantic payload.
"""

import json
import re
from typing import List, Dict, Any, Tuple
import torch
from transformers import PreTrainedTokenizerBase


# Curated Enterprise Structured Extraction Calibration Corpus
CALIBRATION_RECORDS = [
    {
        "domain": "Financial Transactions",
        "context": "Payment notification: On Oct 21, user USR-8812 processed a domestic wire transfer of $14,250.00 USD to recipient vendor Apex Logistics LLC. Transaction reference is WTRX-90142. Risk assessment completed: cleared with no compliance flags.",
        "schema_instruction": "Extract the transaction details into JSON conforming to schema: {transaction_id: str, user_id: str, amount: float, currency: str, recipient: str, is_flagged: bool}.",
        "target_json": {
            "transaction_id": "WTRX-90142",
            "user_id": "USR-8812",
            "amount": 14250.00,
            "currency": "USD",
            "recipient": "Apex Logistics LLC",
            "is_flagged": False
        }
    },
    {
        "domain": "DevOps Incident Response",
        "context": "ALERT: Service 'auth-gateway' triggered HTTP 504 gateway timeout on node k8s-node-us-east-4a at 14:02:11 UTC. Severity level set to P1-CRITICAL. Root cause identified as Redis cluster replica desync. Immediate manual failover required.",
        "schema_instruction": "Extract the incident alert into JSON with schema: {service: str, error_code: int, severity: str, node_id: str, action_required: str, automated_restart: bool}.",
        "target_json": {
            "service": "auth-gateway",
            "error_code": 504,
            "severity": "P1-CRITICAL",
            "node_id": "k8s-node-us-east-4a",
            "action_required": "manual failover",
            "automated_restart": False
        }
    },
    {
        "domain": "Clinical Assessment",
        "context": "Clinical note: Patient MRN-44910 presented with acute persistent bronchospasm and dyspnea. Oxygen saturation measured at 91%. Diagnosis confirmed as Acute Exacerbation of Asthma (ICD-10 J45.901). Prescribed 50mg Prednisone for 5 days. Follow-up consultation scheduled in 7 days.",
        "schema_instruction": "Extract clinical record into JSON conforming to: {patient_mrn: str, diagnosis: str, icd_code: str, spo2_percent: int, follow_up_days: int, admission_required: bool}.",
        "target_json": {
            "patient_mrn": "MRN-44910",
            "diagnosis": "Acute Exacerbation of Asthma",
            "icd_code": "J45.901",
            "spo2_percent": 91,
            "follow_up_days": 7,
            "admission_required": False
        }
    },
    {
        "domain": "E-Commerce Logistics",
        "context": "Shipment update: Consignment PKG-78234 containing 3 units of Industrial Sensor Pod (SKU: ISP-09) shipped via Courier FedEx Priority. Total weight 4.8 kg. Estimated arrival date is 2026-10-25. Signature required upon delivery.",
        "schema_instruction": "Extract shipment dispatch into JSON: {package_id: str, sku: str, quantity: int, carrier: str, weight_kg: float, signature_required: bool}.",
        "target_json": {
            "package_id": "PKG-78234",
            "sku": "ISP-09",
            "quantity": 3,
            "carrier": "FedEx Priority",
            "weight_kg": 4.8,
            "signature_required": True
        }
    },
    {
        "domain": "Customer Support Dispatch",
        "context": "Inbound ticket #TCK-5521 from Enterprise customer Quantum Dynamics. The customer reports API rate limiting errors blocking production workloads. Priority tagged as HIGH. Assigned engineer is Sarah Chen.",
        "schema_instruction": "Extract ticket data into JSON: {ticket_id: str, client_name: str, issue_category: str, priority: str, assigned_to: str, escalated: bool}.",
        "target_json": {
            "ticket_id": "TCK-5521",
            "client_name": "Quantum Dynamics",
            "issue_category": "API rate limiting",
            "priority": "HIGH",
            "assigned_to": "Sarah Chen",
            "escalated": True
        }
    },
    {
        "domain": "Cybersecurity Audit",
        "context": "Firewall audit log: Ingress connection from IP 198.51.100.42 targeting port 22 (SSH) blocked by rule SEC-R-401. 14 repeated failed handshake attempts detected within 30 seconds. Threat classification: Brute Force Attempt. IP placed on temporary blocklist.",
        "schema_instruction": "Extract firewall event into JSON: {source_ip: str, target_port: int, rule_id: str, attack_type: str, attempts: int, is_blocked: bool}.",
        "target_json": {
            "source_ip": "198.51.100.42",
            "target_port": 22,
            "rule_id": "SEC-R-401",
            "attack_type": "Brute Force Attempt",
            "attempts": 14,
            "is_blocked": True
        }
    },
    {
        "domain": "HR Employee Records",
        "context": "Onboarding confirmation: Employee EMP-1093, Marcus Vance, joining department 'Data Engineering' as Senior Distributed Systems Engineer. Work location: Remote (Seattle, WA). Reporting manager is Dr. Emily Thorne. Clearance level 2 verified.",
        "schema_instruction": "Extract employee onboarding into JSON: {emp_id: str, full_name: str, department: str, role: str, location_type: str, clearance_level: int}.",
        "target_json": {
            "emp_id": "EMP-1093",
            "full_name": "Marcus Vance",
            "department": "Data Engineering",
            "role": "Senior Distributed Systems Engineer",
            "location_type": "Remote",
            "clearance_level": 2
        }
    },
    {
        "domain": "Supply Chain Inventory",
        "context": "Warehouse audit report: Warehouse WH-CHICAGO-2 verified stock for item Part-BrakeAssembly-42. Current on-hand quantity is 840 units. Minimum safe buffer is 300 units. Unit cost is $112.50 USD. Reorder status currently NOT required.",
        "schema_instruction": "Extract stock status into JSON: {warehouse_id: str, part_number: str, on_hand: int, safety_buffer: int, unit_cost: float, reorder_needed: bool}.",
        "target_json": {
            "warehouse_id": "WH-CHICAGO-2",
            "part_number": "Part-BrakeAssembly-42",
            "on_hand": 840,
            "safety_buffer": 300,
            "unit_cost": 112.50,
            "reorder_needed": False
        }
    }
]

# Separate Unseen Evaluation Records for Zero-Shot Benchmarking
EVALUATION_RECORDS = [
    {
        "domain": "Legal Contract Extraction",
        "context": "Contract summary: Service agreement SA-2026-99 between CloudScale Inc (Vendor) and Global Corp (Client). Contract effective starting 2026-11-01 for an initial term of 24 months. Total contract valuation is $360,000.00. Automatic renewal clause is active.",
        "schema_instruction": "Extract contract parameters into JSON: {contract_id: str, vendor: str, client: str, term_months: int, total_value: float, auto_renewal: bool}.",
        "target_json": {
            "contract_id": "SA-2026-99",
            "vendor": "CloudScale Inc",
            "client": "Global Corp",
            "term_months": 24,
            "total_value": 360000.00,
            "auto_renewal": True
        }
    },
    {
        "domain": "Cloud Infrastructure Billing",
        "context": "Billing invoice: Account ACC-4102 billed for period 2026-09. Total compute consumption 4,820 vCPU-hours on AWS us-east-1. Total invoice charges $1,446.00 USD. Invoice reference INV-88219. Payment status: PENDING_SETTLEMENT.",
        "schema_instruction": "Extract billing record into JSON: {invoice_id: str, account_id: str, vcpu_hours: int, total_charge: float, currency: str, status: str}.",
        "target_json": {
            "invoice_id": "INV-88219",
            "account_id": "ACC-4102",
            "vcpu_hours": 4820,
            "total_charge": 1446.00,
            "currency": "USD",
            "status": "PENDING_SETTLEMENT"
        }
    },
    {
        "domain": "IoT Environmental Monitoring",
        "context": "Telemetry beacon from Node SENSOR-ENV-104 located in Greenhouse 4. Temperature measured at 28.4 C, relative humidity at 67.2%, CO2 concentration at 780 ppm. Battery reserve at 88%. Status operational: NORMAL.",
        "schema_instruction": "Extract telemetry into JSON: {sensor_id: str, location: str, temperature_c: float, humidity_pct: float, co2_ppm: int, battery_pct: int}.",
        "target_json": {
            "sensor_id": "SENSOR-ENV-104",
            "location": "Greenhouse 4",
            "temperature_c": 28.4,
            "humidity_pct": 67.2,
            "co2_ppm": 780,
            "battery_pct": 88
        }
    },
    {
        "domain": "Database Migration Job",
        "context": "ETL pipeline migration: Job ID ETL-MIG-771 transferred table 'customer_orders' from MySQL 8.0 to Snowflake Data Cloud. Total rows processed 1,250,000 with 0 corrupted rows. Duration was 14.5 minutes. Exit code 0.",
        "schema_instruction": "Extract ETL job record into JSON: {job_id: str, table_name: str, rows_migrated: int, errors: int, duration_mins: float, success: bool}.",
        "target_json": {
            "job_id": "ETL-MIG-771",
            "table_name": "customer_orders",
            "rows_migrated": 1250000,
            "errors": 0,
            "duration_mins": 14.5,
            "success": True
        }
    },
    {
        "domain": "Healthcare Laboratory Result",
        "context": "Lab report LAB-2026-310 for patient PT-7742: Hemoglobin A1c measured at 6.4%, fasting glucose 118 mg/dL. Specimen collected at Central Diagnostics. Critical alert flag: false.",
        "schema_instruction": "Extract lab result into JSON: {lab_id: str, patient_id: str, hba1c_pct: float, glucose_mg_dl: int, facility: str, critical_flag: bool}.",
        "target_json": {
            "lab_id": "LAB-2026-310",
            "patient_id": "PT-7742",
            "hba1c_pct": 6.4,
            "glucose_mg_dl": 118,
            "facility": "Central Diagnostics",
            "critical_flag": False
        }
    },
    {
        "domain": "Energy Grid Telemetry",
        "context": "Substation SUB-NORTH-09 reported load 412.5 MW at 50.02 Hz frequency. Transformer temperature 61 C. Breaker status: CLOSED. Overload warning inactive.",
        "schema_instruction": "Extract substation telemetry into JSON: {substation_id: str, load_mw: float, frequency_hz: float, temp_c: int, breaker_status: str, overload_warning: bool}.",
        "target_json": {
            "substation_id": "SUB-NORTH-09",
            "load_mw": 412.5,
            "frequency_hz": 50.02,
            "temp_c": 61,
            "breaker_status": "CLOSED",
            "overload_warning": False
        }
    },
    {
        "domain": "Insurance Claim Intake",
        "context": "Claim CLM-90412 filed by policyholder POL-5581 for auto collision damage. Estimated repair cost $4,850.00 USD. Deductible $500.00. Adjuster assigned: David Kim. Fraud review required: false.",
        "schema_instruction": "Extract claim into JSON: {claim_id: str, policy_id: str, repair_cost: float, deductible: float, adjuster: str, fraud_review: bool}.",
        "target_json": {
            "claim_id": "CLM-90412",
            "policy_id": "POL-5581",
            "repair_cost": 4850.00,
            "deductible": 500.00,
            "adjuster": "David Kim",
            "fraud_review": False
        }
    },
    {
        "domain": "Aviation Flight Dispatch",
        "context": "Flight dispatch FLT-882 departing SFO for JFK. Aircraft tail N841VA, fuel loaded 48,200 lbs, planned cruise altitude 37000 ft. De-icing required: false.",
        "schema_instruction": "Extract flight dispatch into JSON: {flight_id: str, origin: str, destination: str, tail_number: str, fuel_lbs: int, deicing_required: bool}.",
        "target_json": {
            "flight_id": "FLT-882",
            "origin": "SFO",
            "destination": "JFK",
            "tail_number": "N841VA",
            "fuel_lbs": 48200,
            "deicing_required": False
        }
    },
    {
        "domain": "Retail POS Settlement",
        "context": "Store STR-204 end-of-day settlement batch POS-7719: 342 transactions totaling $18,940.50 USD. Cash variance $0.00. Terminal offline count: 0. Batch reconciled: true.",
        "schema_instruction": "Extract POS batch into JSON: {store_id: str, batch_id: str, tx_count: int, total_usd: float, offline_terminals: int, reconciled: bool}.",
        "target_json": {
            "store_id": "STR-204",
            "batch_id": "POS-7719",
            "tx_count": 342,
            "total_usd": 18940.50,
            "offline_terminals": 0,
            "reconciled": True
        }
    },
    {
        "domain": "Telecom SLA Incident",
        "context": "Backbone link LNK-DAL-ORD experienced 18.4 ms latency spike and 0.02% packet loss for 6 minutes. Ticket NET-4421 opened. SLA breach threshold exceeded: false.",
        "schema_instruction": "Extract network incident into JSON: {link_id: str, ticket_id: str, latency_ms: float, packet_loss_pct: float, duration_mins: int, sla_breached: bool}.",
        "target_json": {
            "link_id": "LNK-DAL-ORD",
            "ticket_id": "NET-4421",
            "latency_ms": 18.4,
            "packet_loss_pct": 0.02,
            "duration_mins": 6,
            "sla_breached": False
        }
    },
    {
        "domain": "Manufacturing QA Batch",
        "context": "Assembly line LINE-B3 produced 1,500 units of Rotor-X9 in lot LOT-2026-88. Defect count: 3 units. Yield rate 99.8%. Lot released for shipment: true.",
        "schema_instruction": "Extract QA lot into JSON: {line_id: str, lot_id: str, units_produced: int, defect_count: int, yield_pct: float, released: bool}.",
        "target_json": {
            "line_id": "LINE-B3",
            "lot_id": "LOT-2026-88",
            "units_produced": 1500,
            "defect_count": 3,
            "yield_pct": 99.8,
            "released": True
        }
    },
    {
        "domain": "Maritime Port Manifest",
        "context": "Vessel MV-PACIFIC-STAR docked at Berth B-12 in Port of Rotterdam. Container count: 1,420 TEU. Manifest ID MAN-9920. Customs hold: true.",
        "schema_instruction": "Extract port manifest into JSON: {vessel_name: str, berth: str, port: str, container_teu: int, manifest_id: str, customs_hold: bool}.",
        "target_json": {
            "vessel_name": "MV-PACIFIC-STAR",
            "berth": "B-12",
            "port": "Port of Rotterdam",
            "container_teu": 1420,
            "manifest_id": "MAN-9920",
            "customs_hold": True
        }
    }
]


def format_prompt(context: str, instruction: str) -> str:
    """Standardizes user prompt formatting with system instruction."""
    return (
        f"<|im_start|>system\n"
        f"You are a deterministic enterprise structured data extraction engine. "
        f"Extract information strictly conforming to the requested JSON schema. "
        f"Output pure, valid JSON with no conversational text or markdown code blocks.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Context:\n{context}\n\n"
        f"Instruction:\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def identify_operational_boundary_tokens_detailed(
    tokenizer: PreTrainedTokenizerBase,
    target_json_str: str,
    boundary_weight: float = 6.0,
    content_weight: float = 1.0,
) -> Tuple[List[int], List[float], List[Dict[str, Any]]]:
    """
    Tokenizes target_json_str and returns (token_ids, weights, token_details)
    with explicit token categories for Stage 1 observability.
    """
    token_ids = tokenizer.encode(target_json_str, add_special_tokens=False)
    weights: List[float] = []
    details: List[Dict[str, Any]] = []

    try:
        parsed = json.loads(target_json_str)
        schema_keys = set(parsed.keys())
    except Exception:
        schema_keys = set(re.findall(r'"([a-zA-Z0-9_-]+)":', target_json_str))

    syntax_chars = set('{}[]:," \n\t')

    for pos, tid in enumerate(token_ids):
        tok_str = tokenizer.decode([tid])
        tok_clean = tok_str.strip()

        if all(c in syntax_chars for c in tok_str):
            cat = "structural_delimiter"
            w = float(boundary_weight)
        elif any(k in tok_clean for k in schema_keys) or tok_clean in {"true", "false", "null"}:
            cat = "schema_key_or_literal"
            w = float(boundary_weight)
        elif re.match(r'^[0-9.]+$', tok_clean):
            cat = "numeric"
            w = float(boundary_weight * 0.7)
        else:
            cat = "semantic_payload"
            w = float(content_weight)

        weights.append(w)
        details.append({
            "position": pos,
            "token_id": int(tid),
            "token_text": tok_str,
            "category": cat,
            "weight": w,
        })

    return token_ids, weights, details


def inspect_calibration_record(
    tokenizer: PreTrainedTokenizerBase,
    item: Dict[str, Any],
    boundary_weight: float = 6.0,
    content_weight: float = 1.0,
    max_length: int = 512,
) -> Dict[str, Any]:
    """Produces a full Stage 1 inspection payload for a single calibration record."""
    prompt_text = format_prompt(item["context"], item["schema_instruction"])
    json_target_str = json.dumps(item["target_json"], indent=2)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    _, _, comp_details = identify_operational_boundary_tokens_detailed(
        tokenizer, json_target_str, boundary_weight=boundary_weight, content_weight=content_weight
    )
    prompt_tokens_count = min(len(prompt_ids), max_length)
    remaining = max(0, max_length - prompt_tokens_count)
    comp_details_trunc = comp_details[:remaining]

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        comp_details_trunc.append({
            "position": len(comp_details_trunc),
            "token_id": int(eos_id),
            "token_text": "<eos>",
            "category": "eos_termination",
            "weight": float(boundary_weight),
        })
        comp_details_trunc = comp_details_trunc[:remaining]

    category_counts: Dict[str, int] = {
        "ignored_prompt": prompt_tokens_count,
        "structural_delimiter": 0,
        "schema_key_or_literal": 0,
        "numeric": 0,
        "semantic_payload": 0,
        "eos_termination": 0,
    }
    for d in comp_details_trunc:
        category_counts[d["category"]] = category_counts.get(d["category"], 0) + 1

    return {
        "domain": item.get("domain", "Unknown"),
        "prompt_text": prompt_text,
        "target_json_str": json_target_str,
        "prompt_token_count": prompt_tokens_count,
        "completion_token_count": len(comp_details_trunc),
        "category_counts": category_counts,
        "completion_tokens": comp_details_trunc,
    }


def compute_dataset_hash(records: List[Dict[str, Any]]) -> str:
    """Computes a deterministic SHA-256 hash over calibration/evaluation records."""
    import hashlib
    payload = json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def identify_operational_boundary_tokens(
    tokenizer: PreTrainedTokenizerBase,
    target_json_str: str,
    boundary_weight: float = 6.0,
    content_weight: float = 1.0,
) -> Tuple[List[int], List[float]]:
    """
    Tokenizes the target JSON string and calculates the operational boundary weight
    for each token position.
    
    Operational Boundary Tokens:
    - Structural delimiters: {, }, [, ], :, ,, \n, "
    - JSON Schema Key names (e.g., "transaction_id", "severity")
    - Literal tokens: true, false, null, numbers
    
    Semantic Payload Tokens:
    - User entity values (e.g., "Apex Logistics LLC", "Sarah Chen")
    """
    token_ids = tokenizer.encode(target_json_str, add_special_tokens=False)
    weights = []
    
    # Identify schema keys in target string
    try:
        parsed = json.loads(target_json_str)
        schema_keys = set(parsed.keys())
    except Exception:
        schema_keys = set(re.findall(r'"([a-zA-Z0-9_-]+)":', target_json_str))

    syntax_chars = set('{}[]:," \n\t')

    for tid in token_ids:
        tok_str = tokenizer.decode([tid])
        tok_clean = tok_str.strip()
        
        # Check if token is purely structural delimiter
        if all(c in syntax_chars for c in tok_str):
            weights.append(boundary_weight)
        # Check if token is part of a schema key or literal keyword
        elif any(k in tok_clean for k in schema_keys) or tok_clean in {"true", "false", "null"}:
            weights.append(boundary_weight)
        # Check if token is numeric
        elif re.match(r'^[0-9.]+$', tok_clean):
            weights.append(boundary_weight * 0.7)
        else:
            # Semantic content token
            weights.append(content_weight)
            
    return token_ids, weights


def build_calibration_dataset(
    tokenizer: PreTrainedTokenizerBase,
    records: List[Dict[str, Any]] = None,
    max_length: int = 512,
    boundary_weight: float = 6.0,
    content_weight: float = 1.0,
) -> List[Dict[str, torch.Tensor]]:
    """
    Encodes full prompts and completions into batchable PyTorch tensors with:
    - input_ids: full sequence
    - attention_mask: 1 for active tokens, 0 for pad
    - labels: -100 for prompt tokens, token_ids for JSON completion
    - operational_weights: 0.0 for prompt tokens, boundary/content weights for JSON completion
    """
    if records is None:
        records = CALIBRATION_RECORDS

    dataset = []

    for item in records:
        prompt_text = format_prompt(item["context"], item["schema_instruction"])
        json_target_str = json.dumps(item["target_json"], indent=2)

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        completion_ids, op_weights = identify_operational_boundary_tokens(
            tokenizer,
            json_target_str,
            boundary_weight=boundary_weight,
            content_weight=content_weight,
        )

        # Append EOS termination token with boundary weight so Fisher profiling
        # preserves end-of-sequence stopping after the closing JSON brace.
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            completion_ids = list(completion_ids) + [int(eos_id)]
            op_weights = list(op_weights) + [float(boundary_weight)]

        # Full sequence
        input_ids = prompt_ids + completion_ids
        # Autoregressive labels: ignore prompt with -100
        labels = [-100] * len(prompt_ids) + completion_ids
        # Operational weights: 0.0 for prompt, exact weights for completion
        full_op_weights = [0.0] * len(prompt_ids) + op_weights

        # Truncate if exceeding max length
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            full_op_weights = full_op_weights[:max_length]

        dataset.append({
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "operational_weights": torch.tensor(full_op_weights, dtype=torch.float32),
            "metadata": item,
        })

    return dataset


if __name__ == "__main__":
    from transformers import AutoTokenizer
    print("Testing operational boundary token masking on sample record...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    ds = build_calibration_dataset(tok, records=CALIBRATION_RECORDS[:2])
    
    sample = ds[0]
    print(f"Sequence length: {len(sample['input_ids'])}")
    print(f"Non-masked operational tokens: {(sample['operational_weights'] > 0).sum().item()}")
    print(f"Max operational boundary weight: {sample['operational_weights'].max().item()}")
    print("Calibration dataset generation test passed successfully!")
