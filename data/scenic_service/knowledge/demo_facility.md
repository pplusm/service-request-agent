# Demo facility guidance

> **Demo-only data.** This document is created for the `service-request-agent`
> project. It is not a real scenic-area service policy, does not assign a real
> handling role, and does not provide a real service-level commitment.

- Source ID: `demo_facility_001`
- Scenario: `scenic_service`
- Intended event type: `facility_fault`
- Citation path: `data/scenic_service/knowledge/demo_facility.md`
- Retrieval keywords: `卫生间没水`, `洗手间无水`, `指示牌损坏`, `照明故障`, `路灯不亮`

## Demonstration scope

This source may be retrieved for text that describes a non-emergency facility
issue, for example a restroom without water, a damaged sign, or a lighting
problem. It is only a source for a demonstration recommendation.

Chinese example phrases: `卫生间没水`, `指示牌损坏`, `照明故障`.

## Demonstration recommendation

When the request includes a location and does not indicate an immediate safety
or health risk, the Agent may create a **demo maintenance follow-up suggestion**.
The returned result must cite `demo_facility_001` and mark the suggestion as a
demo action. A person must confirm any real-world follow-up.

## Review boundaries

The Agent must require human review when any of the following is true:

- The location or another required detail is missing.
- The text indicates a possible injury, a safety concern, or another high-risk
  situation.
- The request cannot be confidently classified.
- This source does not actually support the request.

## Explicit exclusions

This document does not define real service rules, staff roles, response times,
or accuracy claims. It must not be used as evidence for an automatic real-world
dispatch or guarantee.
