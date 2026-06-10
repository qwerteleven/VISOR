# Security Policy

## 1. Network-Level Protection (Macro Attacks)

To mitigate the majority of internet-based attacks, the infrastructure must rely on an internet provider or edge layer that enforces:

- **Rate limiting per IP** to throttle abusive or automated traffic.
- **Botnet firewall** to block known malicious IP ranges and coordinated distributed sources.
- **Suspicious traffic detection** to identify and drop anomalous request patterns before they reach application servers.
- **Content delivery protection** (e.g. via a CDN with DDoS mitigation) to absorb and deflect volumetric attacks.

This layer is the first line of defense against mass/automated attacks.

---

## 2. Input Validation (Individual / Targeted Attacks)

All input originating from the client side must be strictly validated server-side before processing. This includes:

- **Camera frames** received from video streams — validate format, size, integrity, and source.
- **WebSocket messages** — validate schema, payload size, message rate, and authentication context.
- **Passwords and access keys** submitted by users — enforce strength rules, sanitize inputs, and never trust client-side pre-validation.
- **All other client input** — apply allowlisting, type checking, length limits, and encoding validation without exception.

Never trust client input implicitly. Validate everything.

---

## 3. Internal Service Communication Policy

The default policy for all internal server-to-server connections must be **REJECT** — no response unless explicitly authorized.

- Only the minimum required connections between services should be opened and documented.
- All inter-service communication must be authenticated using **unique certificates per service pair** (mutual TLS or equivalent).
- No service should accept connections from another unless a specific, approved rule permits it.

Principle: deny by default, permit by exception.

---

## 4. Credential and Certificate Lifecycle

Passwords and certificates must follow a strict lifecycle policy:

- All credentials and certificates must have a **realistic and enforced expiration date** — no indefinite validity.
- **Credential reuse is prohibited** — each secret, password, and certificate must be unique across services and rotations.
- Rotation schedules must be defined and automated where possible.

---

## 5. Server Hardening and Access Control

To reduce exposure in the event of a breach:

- **Code obfuscation** must be applied on production servers to increase the cost of reverse engineering.
- **Worker/employee personal keys** must follow the principle of least privilege — each key grants access only to the environments strictly required for that role.
- Production, staging, and development environments must have fully isolated access scopes.

---

## 6. Code Review and Merge Authorization

All changes to the codebase must go through a controlled authorization process:

- **All pushes and code merges must be approved by a principal** (lead engineer or equivalent) before being accepted.
- For **protected lines** (release branches, main/production branches), an additional security gate applies:
  - The code must pass a **security checkpoint** as a final condition before merge.
  - This checkpoint requires the change to be **tested at least once in a clean, isolated branch** — no shared state, no pre-existing environment artifacts.
  - Only after this verified clean-branch test may the change be promoted to the protected line.

---

## 7. HTTPS Proxy and API Security

All external connections to internal services are routed exclusively through a **Python HTTPS server acting as a reverse proxy**. No internal service is directly exposed to the outside.

- The proxy must enforce **TLS on all incoming connections** — plain HTTP must be rejected or permanently redirected to HTTPS.
- The proxy is responsible for **terminating TLS**, validating requests, and forwarding only authenticated, well-formed traffic to internal services.
- **CORS policy** must be explicitly defined on the proxy — only approved origins are permitted; wildcard `*` is forbidden in production.
- All **API endpoints must require authentication** — unauthenticated requests must return `401` and be logged.
- The proxy must enforce **per-route authorization** — a valid token does not imply access to all routes.

---

## 8. Authentication and Session Management

User authentication is token-based. The following rules apply:

- **Tokens must be signed** using a strong algorithm (e.g. RS256 or ES256) and verified on every request at the proxy level.
- Tokens must have a **short expiration (TTL)** — long-lived tokens are prohibited without a refresh mechanism.
- **Refresh tokens**, if used, must be rotated on each use and invalidated on logout.
- Tokens must be **invalidated server-side** on logout or on detected anomaly — stateless invalidation alone is insufficient.
- **MFA (Multi-Factor Authentication)** is required for all admin and worker accounts accessing production environments.
- Session inactivity must trigger **automatic token expiration**.
- No token or secret may be stored in client-side code, localStorage, or unprotected cookies — use `HttpOnly`, `Secure`, `SameSite=Strict` cookie flags where applicable.

---

## 9. Audit Logging and Monitoring

All user authentication events and relevant actions must be logged to a **dedicated PostgreSQL history database**.

- Logs must capture at minimum: timestamp, user identity, action performed, source IP, and result (success/failure).
- Logs must be **append-only** — no process should have permission to modify or delete audit records.
- **Failed authentication attempts** must trigger alerts after a defined threshold (e.g. 5 consecutive failures).
- An **intrusion detection** mechanism must monitor for anomalous patterns: unusual access hours, geographic anomalies, privilege escalation attempts.
- Log access must be restricted to authorized personnel only and itself be audited.
- **Incident response plan**: a documented procedure must exist defining steps to take upon confirmed breach — including isolation, notification, forensics, and recovery.

---

## 10. Encryption at Rest

All persistent data must be encrypted at rest:

- **Internal server disks** use **LUKS (Linux Unified Key Setup)** full-disk encryption. Keys must be managed securely and not stored on the same disk.
- **Personal development machines on Windows** use **BitLocker** for full-disk encryption. Recovery keys must be stored in a secure, centralized location — not locally.
- **Database backups** (PostgreSQL) must also be encrypted before storage and during transfer.
- Unencrypted storage of sensitive data — even temporarily — is prohibited.

---

## 11. Data Classification

Data classification levels and handling rules are defined in **`legal_regulation.MD`**. All security controls must be applied in accordance with the classification assigned to each data type.

Any new data type introduced into the system must be classified before being stored or processed.

---

## 12. Vulnerability Management and Supply Chain

- **Dependency scanning** must be integrated into the CI/CD pipeline — all third-party libraries must be checked for known CVEs before merge.
- **SAST (Static Analysis)** and **DAST (Dynamic Analysis)** tools must run automatically on protected branch merges.
- A **penetration testing** schedule must be defined — at minimum annually, and after any major infrastructure change.
- **No hardcoded secrets** are permitted in source code — all credentials must be injected via a secrets management system (e.g. HashiCorp Vault, AWS Secrets Manager, or equivalent). Violations must block the merge.

---

## 13. Operational Security (Human Factors)

- **Secrets management**: no passwords, API keys, or certificates may be stored in plain text files, chat logs, emails, or version control.
- **Employee offboarding**: upon departure, all access keys, tokens, VPN credentials, and certificates belonging to the employee must be revoked within 24 hours.
- **Security training**: all developers and operators must complete security awareness training relevant to their role before accessing production systems.
- **Least privilege principle**: each worker's credentials and keys grant access only to the specific environments and resources required for their current role — access is reviewed periodically and revoked when no longer needed.