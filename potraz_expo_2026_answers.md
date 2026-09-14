# POTRAZ Innovation Expo 2026 — Form Answers (OneGov AI)

**Category:** Seed Level

---

**State the name of solution to be exhibited**

OneGov AI

---

**Briefly narrate the problem(s) the solution addresses**

Zimbabwe's laws, statutory instruments, gazettes and court judgments are scattered across multiple websites, published as scanned PDFs, and written in dense legal language. Citizens, SMEs and even legal practitioners struggle to find what the law currently says on any topic, and professional legal advice is unaffordable for most. OneGov AI solves this by unifying over 15,000 legal instruments and 4,900+ court judgments into one searchable, conversational platform that answers legal questions in plain language, with citations to the actual law and links to the original documents.

---

**Describe the social impact the solution brings to the Community**

OneGov AI democratises access to justice. A vendor in Mbare, a farmer in Gokwe, or a startup founder in Bulawayo can ask "what licence do I need?" or "what are my rights as a tenant?" and get an accurate, cited answer in seconds — free — instead of paying for a consultation or going without. It empowers citizens to know their constitutional rights, helps SMEs comply with regulations and formalise their businesses, and gives rural communities the same legal information access as urban law firms. Since launching in July 2026 the platform has served over 2,700 unique users who have asked hundreds of real legal questions.

---

**Which sector is your solution aligned to?**

ICT / Digital Services (e-Government / LegalTech)
[pick the closest dropdown option]

---

**Select the sub-theme the solution aligns to**

Emerging Technologies and Digital Transformation for Industry Solutions

---

**Describe the technological impact the solution brings to the relevant sector**

OneGov AI introduces retrieval-augmented generation (RAG) to Zimbabwe's legal and public-information sector for the first time: hybrid semantic + keyword search over a vector database of the entire national legal corpus, combined with large language models that answer strictly from retrieved law — eliminating AI hallucination. Automated pipelines collect new documents to keep the corpus current with every new Government Gazette, and a knowledge graph links Acts, Statutory Instruments and judgments so users can trace how laws amend and cite each other. This transforms static, fragmented public records into a live, interactive national resource.

---

**How does the solution utilise open or emerging technologies... ease and speed of development and deployment**

The solution is built almost entirely on open technologies: Python, FastAPI, ChromaDB (open-source vector database), BM25 open-source search and open web standards, with large language model APIs as the reasoning layer. This open stack allowed a single developer to build and deploy the full platform in weeks, not years, at minimal cost. It ships as one Docker container to any cloud, so production deployment takes minutes, and corpus updates run automatically every week with no manual editorial team. The same stack can be redeployed for another jurisdiction, regulator or government department in days of configuration rather than months of development — it is already live in the market at onegovai.app.

---

**Describe the solution to be exhibited and how it links to the theme**

OneGov AI (onegovai.app) is a locally built AI legal assistant that unifies Zimbabwe's entire body of law — the Constitution, 15,000+ Acts and Statutory Instruments, Government Gazettes and 4,900+ court judgments — into one conversational platform. Users ask legal questions in plain language and receive accurate, cited answers linked to the original documents. It links directly to the theme "Harnessing Local Innovation for Inclusive Industrial Transformation and Sustainable Development": it is local innovation, designed and built in Zimbabwe on Zimbabwean legal content; it drives inclusive transformation by giving citizens, SMEs and rural communities the same access to legal and regulatory knowledge previously available only to those who could afford lawyers — lowering the compliance barrier for entrepreneurs entering formal industry; and it supports sustainable development by digitising access to public legal records at near-zero marginal cost, strengthening rule of law, good governance and citizen participation (SDG 16), with a cloud-native footprint requiring no physical infrastructure.

---

**Indicate the province where the solution has been implemented**

Harare (nationally accessible online — users reached in all 10 provinces)

---

**Scalability and Sustainability**

OneGov AI is a lightweight cloud-native solution consuming under 4 GB of memory to serve the entire corpus of 15,000+ legal instruments and 4,900+ judgments. It scales horizontally on demand — no physical infrastructure, minimal energy footprint, and near-zero marginal cost per additional user. Content sustainability is automated: weekly pipelines collect each new Government Gazette from official public sources, so the corpus stays current with no manual editorial team. The architecture can be replicated for other SADC jurisdictions or sector regulators with no redesign, and resource costs grow far slower than the user base, making the solution environmentally and financially sustainable as it scales.

---

**Business Viability and Revenue Model**

OneGov AI operates a freemium model: free limited access for citizens, paid Plus/Pro subscriptions for professionals (lawyers, compliance officers, researchers), and a metered API for institutions integrating legal data. Since launching in July 2026 we have recorded over 2,700 unique visitors and 5,800 site visits, 24 registered users, and our first paying subscribers — early validation of willingness to pay. As the first AI-powered conversational gateway to Zimbabwean legislation and case law, we are creating a new market segment; our addressable market includes the country's roughly 4,000 registered legal practitioners, corporate compliance departments, financial institutions, NGOs and academia. Operating costs are low (single-container cloud hosting plus per-query AI fees), so the model reaches sustainability at modest subscription volumes, with institutional API licensing as the primary growth revenue line.
