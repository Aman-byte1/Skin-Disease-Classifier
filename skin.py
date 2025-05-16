import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import streamlit as st

# Load model and processor
local_path = "./dinov2_skin_disease_model"
image_processor = AutoImageProcessor.from_pretrained(local_path)
model = AutoModelForImageClassification.from_pretrained(local_path)

# Define class names
class_names = [
    'Basal Cell Carcinoma', 'Darier_s Disease', 'Epidermolysis Bullosa Pruriginosa',
    'Hailey-Hailey Disease', 'Herpes Simplex', 'Impetigo', 'Larva Migrans',
    'Leprosy Borderline', 'Leprosy Lepromatous', 'Leprosy Tuberculoid',
    'Lichen Planus', 'Lupus Erythematosus Chronicus Discoides', 'Melanoma',
    'Molluscum Contagiosum', 'Mycosis Fungoides', 'Neurofibromatosis',
    'Papilomatosis Confluentes And Reticulate', 'Pediculosis Capitis',
    'Pityriasis Rosea', 'Porokeratosis Actinic', 'Psoriasis', 'Tinea Corporis',
    'Tinea Nigra', 'Tungiasis', 'actinic keratosis', 'dermatofibroma', 'nevus',
    'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma',
    'vascular lesion'
]

# Detailed descriptions including treatment recommendations and laboratory test hints (10 lines each)
disease_descriptions = {
    "Basal Cell Carcinoma": """1. Basal Cell Carcinoma is a common skin cancer often linked to sun exposure.
2. It typically appears as a pearly or translucent bump.
3. Topical treatments like imiquimod or 5-fluorouracil are commonly recommended.
4. In advanced cases, surgical excision or Mohs micrographic surgery may be indicated.
5. A confirmatory skin biopsy is critical to establish the diagnosis.
6. Imaging tests (e.g., ultrasound) can help assess the lesion’s depth.
7. Blood tests may be done to ensure overall health before surgery.
8. Regular dermatological examinations are important for monitoring.
9. Patients are advised to adopt rigorous sun protection measures.
10. A multidisciplinary evaluation, including laboratory tests, ensures comprehensive care.""",
    
    "Darier_s Disease": """1. Darier’s Disease is a genetic disorder marked by greasy, scaly skin lesions.
2. It commonly affects seborrheic areas like the chest and back.
3. Topical retinoids and corticosteroids help reduce inflammation.
4. Oral retinoids are sometimes considered for more severe cases.
5. A thorough skin examination and family history are essential for diagnosis.
6. A skin biopsy can confirm the typical histopathological features.
7. Blood tests, including liver function tests, may be performed prior to systemic therapy.
8. Photoprotection and regular moisturization help manage symptoms.
9. Consistent follow-up with a dermatologist is advised.
10. Comprehensive laboratory workup assists in ruling out additional systemic involvement.""",
    
    "Epidermolysis Bullosa Pruriginosa": """1. This rare genetic condition leads to itchy and blistering skin lesions.
2. Patients experience chronic, painful lesions that may scar over time.
3. Topical anesthetics and soothing creams help manage pain.
4. Strict wound care and infection prevention measures are essential.
5. A skin biopsy is crucial for confirming the diagnosis.
6. Inflammatory markers and microbial cultures can help identify secondary infections.
7. Maintaining skin moisture with barrier creams is advised.
8. Avoiding trauma to the skin may reduce new lesion formation.
9. Phototherapy may be an option under specialist supervision.
10. Continuous monitoring and supportive care are vital for optimal management.""",
    
    "Hailey-Hailey Disease": """1. Hailey-Hailey Disease is an inherited condition characterized by recurrent blisters in skin folds.
2. It often affects areas subject to friction such as the armpits and groin.
3. Topical steroids and antibiotics reduce inflammation and prevent secondary infections.
4. Systemic immunomodulators may be prescribed in severe cases.
5. A thorough clinical examination and patient history are critical for diagnosis.
6. Skin biopsy helps confirm the typical acantholytic features.
7. Basic blood tests (CBC, inflammatory markers) can be useful for monitoring.
8. Maintaining proper skin hygiene and reducing friction are key.
9. Warm compresses and soothing baths may alleviate discomfort.
10. Regular follow-up ensures treatment adjustments based on lab results and clinical progress.""",
    
    "Herpes Simplex": """1. Herpes Simplex is a viral infection producing painful, recurrent lesions.
2. It typically appears as clusters of blisters near the lips or genital area.
3. Antiviral medications such as acyclovir help reduce outbreak severity.
4. Topical antiviral creams provide localized relief.
5. A detailed clinical examination is fundamental for diagnosis.
6. Laboratory tests including PCR and viral cultures confirm the viral presence.
7. Blood tests can help assess the patient’s immune status.
8. Maintaining good hygiene helps prevent spread.
9. Stress management and a healthy lifestyle may reduce recurrences.
10. Regular follow-ups with a skin specialist ensure proper management and lab monitoring.""",
    
    "Impetigo": """1. Impetigo is a highly contagious bacterial skin infection with red sores forming yellow crusts.
2. It is most common in young children.
3. Topical antibiotics like mupirocin are typically effective.
4. Extensive cases may require oral antibiotics.
5. A clinical examination is generally sufficient for diagnosis.
6. Bacterial cultures can identify the causative organism if needed.
7. Basic blood tests are performed if systemic involvement is suspected.
8. Maintaining strict hygiene is essential to prevent spread.
9. Environmental decontamination is recommended to avoid reinfection.
10. Regular follow-up ensures laboratory parameters remain within normal limits during treatment.""",
    
    "Larva Migrans": """1. Larva Migrans is caused by parasitic larvae migrating under the skin, producing an itchy, winding rash.
2. It is commonly seen in tropical or subtropical regions.
3. Antiparasitic medications like ivermectin are the treatment of choice.
4. Topical anti-itch creams provide symptomatic relief.
5. A clinical evaluation based on lesion appearance and history is typically sufficient.
6. Skin scrapings with KOH preparation may help confirm the diagnosis.
7. A blood test to check for eosinophilia can be supportive.
8. Preventive measures such as proper footwear are advised.
9. Maintaining good personal hygiene helps prevent recurrence.
10. Regular follow-up and laboratory workup help assess treatment response.""",
    
    "Leprosy Borderline": """1. Leprosy Borderline is caused by Mycobacterium leprae and presents with discolored or numb skin lesions.
2. It lies between the tuberculoid and lepromatous forms in severity.
3. Multidrug therapy (including dapsone and rifampicin) is the standard treatment.
4. Topical treatments may alleviate localized symptoms.
5. A skin biopsy along with slit-skin smears is key for diagnosis.
6. Nerve conduction studies assess the extent of nerve involvement.
7. PCR and bacterial index tests help in identifying Mycobacterium leprae.
8. CBC and inflammatory markers are monitored during treatment.
9. Early intervention prevents permanent nerve damage.
10. Regular dermatological and neurological evaluations ensure comprehensive care.""",
    
    "Leprosy Lepromatous": """1. Leprosy Lepromatous is a severe form with widespread skin lesions due to a weak immune response.
2. It requires a multidrug regimen for effective treatment.
3. Topical agents can help relieve local symptoms.
4. Skin biopsy and slit-skin smears are critical for diagnosis.
5. Detailed PCR tests quantify the bacterial load.
6. Nerve conduction studies evaluate neurological involvement.
7. Regular blood tests monitor treatment response.
8. Patient education and counseling are important during therapy.
9. A multidisciplinary approach ensures optimal patient care.
10. Laboratory tests combined with clinical follow-up track disease progression.""",
    
    "Leprosy Tuberculoid": """1. Leprosy Tuberculoid presents with well-defined hypopigmented lesions and sensory loss.
2. It is characterized by a strong immune response to Mycobacterium leprae.
3. Treatment involves multidrug therapy including dapsone and rifampicin.
4. Topical steroids may reduce localized inflammation.
5. A skin biopsy confirms the diagnosis.
6. Slit-skin smears and nerve conduction studies are recommended.
7. Sensory testing further supports the diagnosis.
8. Regular blood tests help monitor treatment efficacy.
9. Patient education on preventing trauma to affected areas is crucial.
10. Consistent follow-up by a skin specialist ensures early detection of complications.""",
    
    "Lichen Planus": """1. Lichen Planus is an inflammatory condition marked by purple, pruritic, polygonal papules.
2. It can affect both the skin and mucous membranes.
3. Topical corticosteroids are usually the first line of treatment.
4. Oral antihistamines help control itching.
5. A skin biopsy confirms the diagnosis.
6. Blood tests—including liver function tests and autoimmune panels—may be indicated.
7. Phototherapy is a treatment option for widespread lesions.
8. Avoidance of known triggers helps reduce flare-ups.
9. Regular dermatologist visits assist in monitoring progression.
10. Comprehensive laboratory evaluations ensure no systemic involvement is overlooked.""",
    
    "Lupus Erythematosus Chronicus Discoides": """1. Lupus Erythematosus Chronicus Discoides presents with chronic discoid lesions that may scar.
2. It is a form of cutaneous lupus with limited systemic involvement.
3. Topical corticosteroids and calcineurin inhibitors are used for skin lesions.
4. Oral antimalarials such as hydroxychloroquine may be prescribed.
5. A skin biopsy differentiates it from other dermatoses.
6. Laboratory tests including ANA, anti-dsDNA, and complement levels are recommended.
7. Photoprotection is crucial to prevent lesion exacerbation.
8. CBC and metabolic panels are monitored during treatment.
9. Detailed immunological studies help tailor therapy.
10. Regular follow-up with laboratory tests ensures optimal long-term management.""",
    
    "Melanoma": """1. Melanoma is an aggressive skin cancer arising from melanocytes.
2. It usually presents as an irregular, asymmetrical pigmented lesion.
3. Early detection through self-examination is crucial.
4. Surgical excision with clear margins is the primary treatment.
5. A skin biopsy with histopathology confirms the diagnosis.
6. Sentinel lymph node biopsy is recommended to assess metastasis.
7. Blood tests such as LDH may help in monitoring advanced cases.
8. Imaging studies (CT/MRI) assist in staging the disease.
9. Patient education on sun protection and skin surveillance is vital.
10. Regular multidisciplinary follow-up, including lab tests, is essential for management.""",
    
    "Molluscum Contagiosum": """1. Molluscum Contagiosum is a viral infection characterized by small, dome-shaped papules.
2. It is common in children and immunocompromised individuals.
3. Cryotherapy and topical agents such as imiquimod are common treatments.
4. Physical removal (e.g., curettage) is often effective.
5. Clinical evaluation is usually sufficient for diagnosis.
6. Laboratory tests are rarely needed unless there is suspicion of an immune deficiency.
7. In recurrent cases, immune status evaluation may be considered.
8. Good hygiene helps limit spread.
9. Follow-up ensures lesions resolve without complications.
10. Periodic evaluation by a skin specialist confirms diagnosis and guides management.""",
    
    "Mycosis Fungoides": """1. Mycosis Fungoides is a type of cutaneous T-cell lymphoma presenting as patches and plaques.
2. It often follows an indolent but progressive course.
3. Topical corticosteroids and phototherapy are common first-line treatments.
4. Systemic therapies may be introduced in advanced stages.
5. A skin biopsy with immunohistochemistry is essential for diagnosis.
6. Flow cytometry and molecular studies determine T-cell clonality.
7. Regular blood tests and LDH measurements help monitor disease activity.
8. Imaging studies are used for accurate disease staging.
9. Patient education on the chronic nature of the condition is important.
10. Multidisciplinary follow-up, including periodic lab tests, is recommended.""",
    
    "Neurofibromatosis": """1. Neurofibromatosis is a genetic disorder that causes benign nerve sheath tumors.
2. It commonly presents with multiple neurofibromas on the skin.
3. Surgical removal may be considered for symptomatic lesions.
4. Topical treatments can help manage localized discomfort.
5. Genetic testing is recommended to confirm the diagnosis.
6. MRI or CT scans assess deeper or plexiform neurofibromas.
7. Regular neurological examinations and skin checks are critical.
8. Routine blood panels help monitor overall health.
9. Counseling and genetic advice support patient management.
10. Continuous follow-up with specialists, along with lab tests, ensures early detection of complications.""",
    
    "Papilomatosis Confluentes And Reticulate": """1. This rare condition presents with confluent papules in a reticulate pattern.
2. It predominantly affects flexural areas with cosmetic concerns.
3. Topical retinoids and corticosteroids may improve skin appearance.
4. Oral retinoids are an option for widespread involvement.
5. A skin biopsy confirms the diagnosis.
6. Basic blood tests (CBC, liver panel) help monitor systemic effects.
7. Regular dermatological evaluations are advised.
8. Imaging studies are rarely required unless deeper involvement is suspected.
9. Patient education regarding treatment adherence is critical.
10. Periodic laboratory tests ensure that therapy remains safe and effective.""",
    
    "Pediculosis Capitis": """1. Pediculosis Capitis (head lice infestation) is marked by intense scalp itching.
2. It is most common in school-aged children.
3. Over-the-counter pediculicides such as permethrin are effective.
4. Manual removal using a fine-toothed comb supports treatment.
5. A clinical examination confirms the diagnosis.
6. Laboratory tests are rarely required for head lice.
7. Regular scalp inspections help prevent reinfestation.
8. Proper hygiene and environmental cleaning are essential.
9. Alternative medications may be considered in resistant cases.
10. Follow-up by a healthcare provider ensures complete eradication.""",
    
    "Pityriasis Rosea": """1. Pityriasis Rosea is a self-limiting rash often preceded by a herald patch.
2. It manifests as widespread, oval pink patches.
3. Topical corticosteroids help alleviate itching.
4. Oral antihistamines can be used for symptomatic relief.
5. A clinical examination is generally sufficient for diagnosis.
6. Basic blood tests like CBC help rule out other conditions.
7. Skin scrapings may be performed if the presentation is atypical.
8. Proper skin care and hydration are advised.
9. Patient education reassures that the condition is self-limiting.
10. Regular follow-up confirms resolution and monitors labs if needed.""",
    
    "Porokeratosis Actinic": """1. Porokeratosis Actinic presents as annular lesions with raised borders on sun-exposed skin.
2. It is considered a precancerous condition.
3. Topical treatments such as 5-fluorouracil or imiquimod are used.
4. Cryotherapy may be applied for localized lesions.
5. A skin biopsy is essential for definitive diagnosis.
6. Dermoscopy aids in evaluating early malignant changes.
7. Laboratory tests are minimal unless atypical features appear.
8. Strict sun protection is strongly recommended.
9. Regular dermatologic assessments monitor lesion changes.
10. Follow-up with lab tests helps ensure no malignant transformation occurs.""",
    
    "Psoriasis": """1. Psoriasis is a chronic autoimmune condition characterized by red, scaly plaques.
2. It commonly affects the scalp, elbows, and knees.
3. Topical corticosteroids and vitamin D analogs are first-line treatments.
4. Systemic agents or biologics may be used for moderate-to-severe cases.
5. A detailed clinical examination and history are critical for diagnosis.
6. Blood tests (CBC, CRP/ESR) help monitor disease activity.
7. A skin biopsy may be considered in atypical presentations.
8. Phototherapy is an effective treatment option.
9. Patient education on triggers and lifestyle modifications is vital.
10. Regular follow-up with laboratory evaluations ensures optimal long-term management.""",
    
    "Tinea Corporis": """1. Tinea Corporis (ringworm) is a fungal infection presenting as ring-shaped, scaly lesions.
2. It is typically diagnosed through clinical examination.
3. Topical antifungals like clotrimazole are usually effective.
4. Oral antifungals may be required for extensive or resistant cases.
5. Skin scrapings with KOH preparation help confirm the diagnosis.
6. Fungal cultures can further support the findings.
7. Basic blood tests are generally not required unless systemic infection is suspected.
8. Maintaining good hygiene is essential to prevent spread.
9. Patient education on cleaning personal items is advised.
10. Follow-up examinations ensure complete resolution and monitor labs if needed.""",
    
    "Tinea Nigra": """1. Tinea Nigra is a superficial fungal infection causing dark, velvety patches, usually on the palms or soles.
2. It is generally benign and asymptomatic.
3. Topical antifungal agents are the treatment of choice.
4. A clinical diagnosis is supported by KOH preparation of skin scrapings.
5. Fungal culture may be performed for confirmation.
6. Laboratory tests are minimal unless secondary infection is suspected.
7. Proper skin hygiene is essential.
8. Differential diagnosis with pigmented lesions is important.
9. Regular follow-up confirms proper treatment response.
10. A skin specialist may perform additional lab tests if the presentation is atypical.""",
    
    "Tungiasis": """1. Tungiasis is caused by the sand flea penetrating the skin, most commonly on the feet.
2. It presents with painful, itchy lesions in tropical regions.
3. The primary treatment involves careful removal of the embedded flea.
4. Topical antiseptics and antibiotics prevent secondary infections.
5. A clinical examination confirms the diagnosis.
6. Laboratory tests are rarely needed unless systemic infection is suspected.
7. Imaging studies are generally unnecessary.
8. Proper foot hygiene and protective footwear are advised.
9. Routine blood tests may be done if systemic symptoms develop.
10. Follow-up with a skin specialist ensures complete recovery and treatment safety.""",
    
    "actinic keratosis": """1. Actinic Keratosis is a precancerous skin condition from chronic sun exposure.
2. It appears as rough, scaly patches on sun-exposed areas.
3. Topical treatments like 5-fluorouracil or imiquimod are used.
4. Cryotherapy is effective for isolated lesions.
5. A skin biopsy is warranted if the lesion appears suspicious.
6. Dermoscopy assists in early detection of malignant changes.
7. Basic blood tests are generally normal unless otherwise indicated.
8. Strict sun protection is essential.
9. Imaging studies may be recommended in uncertain cases.
10. Regular follow-up with lab tests ensures early detection of progression.""",
    
    "dermatofibroma": """1. Dermatofibroma is a benign fibrous nodule often found on the legs.
2. It is usually asymptomatic but may cause mild discomfort.
3. Observation is typically recommended for stable lesions.
4. Surgical excision is an option if symptoms worsen.
5. A skin biopsy confirms the diagnosis when in doubt.
6. Dermoscopy supports differentiation from malignant lesions.
7. Laboratory tests are generally not required.
8. Patient reassurance and education are important.
9. Regular follow-up helps monitor any changes.
10. A thorough clinical examination ensures no further intervention is needed.""",
    
    "nevus": """1. A nevus (mole) is a benign collection of melanocytes.
2. It typically appears as a uniform pigmented spot.
3. Observation is recommended if the nevus remains stable.
4. Surgical removal is considered if significant changes occur.
5. A skin biopsy rules out melanoma if suspicious.
6. Dermoscopic evaluation supports an accurate diagnosis.
7. Laboratory tests are not typically needed.
8. Regular self-examinations and clinical checks are advised.
9. Patients should practice sun protection to prevent changes.
10. Follow-up visits with a dermatologist ensure early detection of malignancy.""",
    
    "pigmented benign keratosis": """1. Pigmented benign keratosis is a non-cancerous skin lesion with a rough texture.
2. Its pigmented appearance can mimic malignant lesions.
3. Topical treatments and cryotherapy may improve appearance.
4. A thorough clinical examination is necessary.
5. A skin biopsy can be done if the lesion appears atypical.
6. Histopathology confirms its benign nature.
7. Regular dermoscopic monitoring is advised.
8. Patients are counseled on sun protection.
9. Laboratory tests help rule out malignancy.
10. Follow-up with a skin specialist ensures ongoing stability.""",
    
    "seborrheic keratosis": """1. Seborrheic keratosis is a common benign growth appearing as a waxy, brown lesion.
2. It is non-cancerous despite its appearance.
3. Cryotherapy, curettage, or laser therapy can be used for removal.
4. A skin biopsy is considered if there is diagnostic uncertainty.
5. Clinical examination combined with dermoscopy supports the diagnosis.
6. Laboratory tests are rarely required.
7. Patients are advised to monitor for rapid changes.
8. Routine follow-up confirms lesion stability.
9. Patient education reassures the benign nature.
10. Detailed evaluation by a skin specialist confirms proper management.""",
    
    "squamous cell carcinoma": """1. Squamous Cell Carcinoma arises from keratinocytes and is a common skin cancer.
2. It typically presents as a scaly, red nodule or lesion.
3. Surgical excision or Mohs micrographic surgery is the primary treatment.
4. Topical chemotherapeutic agents may be used for superficial lesions.
5. A skin biopsy is necessary for definitive diagnosis.
6. Imaging studies (CT/MRI) assess lesion extent and metastasis.
7. Blood tests including CBC and inflammatory markers are recommended.
8. Strict sun protection is critical.
9. Regular follow-up monitors for potential metastasis.
10. A comprehensive treatment plan includes periodic laboratory tests.""",
    
    "vascular lesion": """1. Vascular lesions include conditions such as hemangiomas and port-wine stains.
2. They present as red to purple discolorations of the skin.
3. Laser therapy and sclerotherapy are common treatment options.
4. Topical treatments may alleviate minor symptoms.
5. A clinical examination is typically sufficient for diagnosis.
6. Doppler ultrasound assesses blood flow within the lesion.
7. Coagulation profiles may be performed as part of lab evaluation.
8. Detailed patient history guides treatment planning.
9. Follow-up ensures treatment efficacy.
10. A multidisciplinary approach, including lab tests, is recommended for optimal management."""
}

# Additional dictionary for specific laboratory tests recommended for each disease
disease_lab_tests = {
    "Basal Cell Carcinoma": """- **Skin Biopsy:** For histopathological confirmation
- **Ultrasound Imaging:** To assess lesion depth
- **Complete Blood Count (CBC):** To evaluate overall health
- **Inflammatory Markers:** (CRP, ESR)""",
    
    "Darier_s Disease": """- **Skin Biopsy:** For diagnostic confirmation
- **Liver Function Tests:** Especially if considering systemic retinoids
- **Complete Blood Count (CBC):** To monitor overall health
- **Autoimmune Panel:** If indicated by clinical history""",
    
    "Epidermolysis Bullosa Pruriginosa": """- **Skin Biopsy:** For definitive diagnosis
- **Inflammatory Marker Tests:** (CRP, ESR) to check for secondary infections
- **Microbial Cultures:** To rule out superinfections""",
    
    "Hailey-Hailey Disease": """- **Skin Biopsy:** To identify characteristic histopathology
- **Complete Blood Count (CBC):** Routine health check
- **Inflammatory Markers:** To assess systemic inflammation""",
    
    "Herpes Simplex": """- **PCR Testing:** For viral DNA detection
- **Viral Culture:** To confirm active infection
- **Complete Blood Count (CBC):** To evaluate immune status""",
    
    "Impetigo": """- **Bacterial Culture:** To identify causative organisms
- **Complete Blood Count (CBC):** If systemic infection is suspected""",
    
    "Larva Migrans": """- **Skin Scrapings/KOH Prep:** To identify parasitic elements
- **Eosinophil Count:** As an indicator of parasitic infection""",
    
    "Leprosy Borderline": """- **Slit-Skin Smear:** For acid-fast bacilli detection
- **Skin Biopsy:** With histopathological examination
- **PCR Testing:** For Mycobacterium leprae
- **Nerve Conduction Studies:** To assess nerve involvement""",
    
    "Leprosy Lepromatous": """- **Slit-Skin Smear:** For bacterial load estimation
- **Skin Biopsy:** To confirm diagnosis
- **PCR Testing:** For Mycobacterium leprae identification
- **Complete Blood Count (CBC):** For monitoring""",
    
    "Leprosy Tuberculoid": """- **Skin Biopsy:** For histopathological analysis
- **Slit-Skin Smear:** For bacterial detection
- **Nerve Conduction Studies:** To evaluate sensory deficits
- **Sensory Testing:** As part of neurological assessment""",
    
    "Lichen Planus": """- **Skin Biopsy:** For definitive diagnosis
- **Liver Function Tests:** To rule out drug-induced causes
- **Autoimmune Panel:** For associated autoimmune conditions""",
    
    "Lupus Erythematosus Chronicus Discoides": """- **ANA and Anti-dsDNA:** For autoimmune screening
- **Complement Levels:** (C3, C4)
- **Skin Biopsy:** For histopathological confirmation""",
    
    "Melanoma": """- **Skin Biopsy:** With histopathological evaluation
- **Sentinel Lymph Node Biopsy:** For staging
- **LDH Levels:** For advanced disease monitoring
- **Complete Blood Count (CBC):** Routine evaluation""",
    
    "Molluscum Contagiosum": """- **Clinical Evaluation:** Primary diagnostic tool
- **Immune Status Assessment:** (if recurrent cases occur)""",
    
    "Mycosis Fungoides": """- **Skin Biopsy with Immunohistochemistry:** For diagnostic confirmation
- **Flow Cytometry:** To assess T-cell clonality
- **LDH Levels and CBC:** For monitoring disease activity""",
    
    "Neurofibromatosis": """- **Genetic Testing:** To confirm diagnosis
- **MRI/CT Scans:** For assessing plexiform neurofibromas
- **Complete Blood Count (CBC):** Routine monitoring""",
    
    "Papilomatosis Confluentes And Reticulate": """- **Skin Biopsy:** For definitive diagnosis
- **Complete Blood Count (CBC):** Routine evaluation
- **Liver Function Tests:** If systemic therapy is considered""",
    
    "Pediculosis Capitis": """- **Clinical Examination:** Primary diagnosis
- *No specific laboratory tests are typically required*""",
    
    "Pityriasis Rosea": """- **Complete Blood Count (CBC):** To exclude other conditions
- **Basic Metabolic Panel:** As part of routine evaluation""",
    
    "Porokeratosis Actinic": """- **Skin Biopsy:** If lesions appear atypical
- **Dermoscopy:** To evaluate lesion characteristics
- **Complete Blood Count (CBC):** Routine test if indicated""",
    
    "Psoriasis": """- **Complete Blood Count (CBC):** For routine health monitoring
- **CRP/ESR:** To assess inflammatory status
- **Liver Function Tests:** If systemic treatments are used""",
    
    "Tinea Corporis": """- **KOH Preparation:** For fungal elements detection
- **Fungal Culture:** To confirm the pathogen
- *CBC is rarely required unless systemic involvement is suspected*""",
    
    "Tinea Nigra": """- **KOH Preparation:** For confirming fungal elements
- **Fungal Culture:** If clinical diagnosis is uncertain""",
    
    "Tungiasis": """- **Clinical Examination:** Primary diagnosis
- **Complete Blood Count (CBC):** If secondary infection is suspected""",
    
    "actinic keratosis": """- **Skin Biopsy:** For lesions with suspicious features
- **Dermoscopy:** To monitor lesion changes
- *CBC is typically normal unless complications arise*""",
    
    "dermatofibroma": """- **Skin Biopsy:** If diagnostic uncertainty exists
- **Dermoscopy:** For lesion evaluation""",
    
    "nevus": """- **Dermoscopy:** To assess for atypical features
- **Skin Biopsy:** If there are significant changes""",
    
    "pigmented benign keratosis": """- **Dermoscopy:** For routine evaluation
- **Skin Biopsy:** If the lesion shows atypical features""",
    
    "seborrheic keratosis": """- **Clinical Evaluation:** Primary diagnostic tool
- **Dermoscopy:** To rule out malignancy
- **Skin Biopsy:** If the diagnosis is uncertain""",
    
    "squamous cell carcinoma": """- **Skin Biopsy:** For histopathological confirmation
- **CT/MRI Scans:** For staging
- **Complete Blood Count (CBC):** Routine evaluation
- **Inflammatory Markers:** (CRP, ESR)""",
    
    "vascular lesion": """- **Doppler Ultrasound:** To assess blood flow
- **Coagulation Profile:** As part of laboratory evaluation
- **Clinical Examination:** Primary diagnostic tool"""
}
# UI Layouttt
# centered is good
st.set_page_config(page_title="የቆዳ በሽታ መለያ", layout="centered")
st.title("🧑‍⚕️ የቆዳ በሽታ መለያ") # Skin Disease Classifier
st.write("ሊኖር የሚችል የቆዳ በሽታን ለመለየት የቆዳ ምስል ይስቀሉ።") # Upload a skin image to identify possible skin diseases

# File uploaderrr
# choose fileee
uploaded_file = st.file_uploader("ምስል ይምረጡ...", type=["jpg", "jpeg", "png"]) # Choose an image...

if uploaded_file is not None:
    # opennn and display image
    # convert to RGB alwaysss
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="የተሰቀለ ምስል", use_column_width=True) # Uploaded Image

    # preprocessss and predicttt
    # no grad needed
    encoding = image_processor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        # get indexxx
        predicted_class_idx = logits.argmax(-1).item()
        # get nameee
        predicted_class_name = class_names[predicted_class_idx]

    # display resultss
    # make it greennn
    st.success(f"🩺 የተተነበየ የቆዳ በሽታ: **{predicted_class_name}**") # Predicted Skin Disease:

    # get and display descriptionnn
    # check if description exists
    description = disease_descriptions.get(predicted_class_name, "ለዚህ ሁኔታ ዝርዝር መግለጫ የለም።") # Detailed description not available for this condition.
    st.markdown(f"### ዝርዝር መግለጫ እና ምክሮች:\n{description}") # Detailed Description and Recommendations:

    # get and display lab testsss
    # lab tests infooo
    lab_tests = disease_lab_tests.get(predicted_class_name, "የላብራቶሪ ምርመራ መረጃ ለዚህ ሁኔታ የለም።") # Laboratory tests information not available for this condition.
    st.markdown(f"### የሚመከሩ የላብራቶሪ ምርመራዎች:\n{lab_tests}") # Recommended Laboratory Tests:

