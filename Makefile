PYTHON ?= python3
DATA_DIR ?= data/raw
BUILD_DIR ?= build
ASSIGNMENT_BUDGET ?= 10
ASSIGNMENT_RADIUS_KM ?= 5
WEAK_QUANTILE ?= 0.4
MAX_SUBJECTS_PER_DOMAIN ?= 8
PYTHONPATH := src
export PYTHONPATH

META_DIR := $(BUILD_DIR)/metadata
INTERIM_DIR := $(BUILD_DIR)/interim
PROCESSED_DIR := $(BUILD_DIR)/processed
REVIEW_DIR := $(BUILD_DIR)/review
SUBJECT_OVERRIDES := config/subject_overrides.csv
SUBJECT_IGNORES := config/subject_ignores.csv

.PHONY: all check-inputs clean rebuild collect-neis geocode-facilities facility-accessibility recommend-greedy recommend-rl

all: \
	$(INTERIM_DIR)/school_master.csv \
	$(PROCESSED_DIR)/neis_subjects.csv \
	$(PROCESSED_DIR)/school_subject_summary.csv \
	$(INTERIM_DIR)/analysis_schools.csv \
	$(PROCESSED_DIR)/nearby_school_accessibility.csv \
	$(INTERIM_DIR)/facilities.csv \
	$(INTERIM_DIR)/joint_curriculum.csv \
	$(PROCESSED_DIR)/joint_curriculum_existing_network.csv \
	$(PROCESSED_DIR)/school_features.csv \
	$(BUILD_DIR)/tables/school_sai_result.csv

.PHONY: subject-review-list review-subjects

check-inputs: $(META_DIR)/input_manifest.csv

$(META_DIR)/input_manifest.csv: scripts/00_check_inputs.py config/inputs.yml
	$(PYTHON) $< --data-dir "$(DATA_DIR)" --out "$@"

$(DATA_DIR)/outputs/raw/neis_school_info_raw.csv $(DATA_DIR)/outputs/raw/neis_his_timetable_raw.csv: \
	scripts/00_collect_neis_raw.py \
	src/coursemap/env.py \
	src/coursemap/io.py
	$(PYTHON) scripts/00_collect_neis_raw.py \
		--data-dir "$(DATA_DIR)" \
		--school-info-out "$(DATA_DIR)/outputs/raw/neis_school_info_raw.csv" \
		--timetable-out "$(DATA_DIR)/outputs/raw/neis_his_timetable_raw.csv" \
		--log-out "$(META_DIR)/neis_timetable_collection_log.csv"

collect-neis: $(DATA_DIR)/outputs/raw/neis_school_info_raw.csv $(DATA_DIR)/outputs/raw/neis_his_timetable_raw.csv

$(INTERIM_DIR)/school_master.csv: \
	scripts/01_prepare_school_master.py \
	$(DATA_DIR)/school_location_20260320.csv \
	$(DATA_DIR)/schoolinfo_2025_daejeon_high_student_class.csv \
	$(DATA_DIR)/schoolinfo_2025_daejeon_high_teacher.csv \
	$(DATA_DIR)/schoolinfo_2025_daejeon_high_school_building.csv \
	$(DATA_DIR)/schoolinfo_2025_daejeon_high_support_facilities.csv \
	$(DATA_DIR)/outputs/raw/neis_school_info_raw.csv
	$(PYTHON) $< --data-dir "$(DATA_DIR)" --out "$@"

$(PROCESSED_DIR)/neis_subjects.csv: \
	scripts/02_prepare_neis_subjects.py \
	src/coursemap/io.py \
	src/coursemap/text.py \
	src/coursemap/subjects.py \
	$(DATA_DIR)/outputs/raw/neis_his_timetable_raw.csv
	$(PYTHON) $< --data-dir "$(DATA_DIR)" --out "$@"

$(PROCESSED_DIR)/school_subject_summary.csv: \
	scripts/05_build_subject_supply.py \
	src/coursemap/io.py \
	src/coursemap/subjects.py \
	$(PROCESSED_DIR)/neis_subjects.csv \
	$(SUBJECT_OVERRIDES) \
	$(SUBJECT_IGNORES)
	$(PYTHON) $< --subjects "$(PROCESSED_DIR)/neis_subjects.csv" \
		--overrides "$(SUBJECT_OVERRIDES)" \
		--ignores "$(SUBJECT_IGNORES)" \
		--out-dir "$(PROCESSED_DIR)" \
		--review-out "$(REVIEW_DIR)/unassigned_subjects.csv"

$(SUBJECT_OVERRIDES):
	mkdir -p "$(dir $@)"
	printf 'subject,standard_subject,domain\n' > "$@"

$(SUBJECT_IGNORES):
	mkdir -p "$(dir $@)"
	printf 'subject,reason\n' > "$@"

subject-review-list: scripts/05_build_subject_supply.py $(PROCESSED_DIR)/neis_subjects.csv $(SUBJECT_OVERRIDES) $(SUBJECT_IGNORES)
	$(PYTHON) $< --subjects "$(PROCESSED_DIR)/neis_subjects.csv" \
		--overrides "$(SUBJECT_OVERRIDES)" \
		--ignores "$(SUBJECT_IGNORES)" \
		--out-dir "$(PROCESSED_DIR)" \
		--review-out "$(REVIEW_DIR)/unassigned_subjects.csv" \
		--allow-unassigned

review-subjects: subject-review-list
	$(PYTHON) scripts/review_subjects.py \
		--review "$(REVIEW_DIR)/unassigned_subjects.csv" \
		--overrides "$(SUBJECT_OVERRIDES)"

$(INTERIM_DIR)/facilities.csv: \
	scripts/03_prepare_facilities.py \
	$(DATA_DIR)/public_library_2025_daejeon_filtered.csv \
	$(DATA_DIR)/youth_training_facilities_daejeon_filtered.csv \
	$(DATA_DIR)/lifelong_education_facilities_daejeon_20260309.csv
	$(PYTHON) $< --data-dir "$(DATA_DIR)" --out "$@"

$(INTERIM_DIR)/facilities_geocoded.csv: \
	scripts/04_geocode_facilities.py \
	src/coursemap/geocode.py \
	src/coursemap/io.py \
	$(INTERIM_DIR)/facilities.csv
	$(PYTHON) $< --facilities "$(INTERIM_DIR)/facilities.csv" \
		--data-dir "$(DATA_DIR)" \
		--out "$@" \
		--log-out "$(BUILD_DIR)/metadata/geocoding_log.csv"

$(PROCESSED_DIR)/facility_accessibility.csv: \
	scripts/05_build_facility_accessibility.py \
	src/coursemap/geo.py \
	src/coursemap/io.py \
	$(INTERIM_DIR)/analysis_schools.csv \
	$(INTERIM_DIR)/facilities_geocoded.csv
	$(PYTHON) $< --schools "$(INTERIM_DIR)/analysis_schools.csv" \
		--facilities "$(INTERIM_DIR)/facilities_geocoded.csv" \
		--out "$@"

geocode-facilities: $(INTERIM_DIR)/facilities_geocoded.csv

facility-accessibility: $(PROCESSED_DIR)/facility_accessibility.csv

recommend-greedy: $(BUILD_DIR)/tables/school_sai_result.csv
	MPLCONFIGDIR=/tmp/matplotlib $(PYTHON) scripts/10_recommend_joint_assignments.py \
		--budget "$(ASSIGNMENT_BUDGET)" \
		--radius-km "$(ASSIGNMENT_RADIUS_KM)" \
		--weak-quantile "$(WEAK_QUANTILE)" \
		--max-subjects-per-domain "$(MAX_SUBJECTS_PER_DOMAIN)"

recommend-rl: $(BUILD_DIR)/tables/school_sai_result.csv
	MPLCONFIGDIR=/tmp/matplotlib $(PYTHON) scripts/11_train_rl_assignments.py \
		--budget "$(ASSIGNMENT_BUDGET)" \
		--radius-km "$(ASSIGNMENT_RADIUS_KM)" \
		--weak-quantile "$(WEAK_QUANTILE)" \
		--max-subjects-per-domain "$(MAX_SUBJECTS_PER_DOMAIN)"

$(INTERIM_DIR)/joint_curriculum.csv: \
	scripts/04_prepare_joint_curriculum.py \
	$(DATA_DIR)/daejeon_joint_curriculum_2025_1st.xlsx \
	$(DATA_DIR)/daejeon_joint_curriculum_2025_2nd.xlsx
	$(PYTHON) $< --data-dir "$(DATA_DIR)" --out "$@"

$(INTERIM_DIR)/analysis_schools.csv: \
	scripts/06_validate_feature_coverage.py \
	src/coursemap/io.py \
	config/blacklists.yml \
	$(INTERIM_DIR)/school_master.csv \
	$(PROCESSED_DIR)/school_subject_summary.csv
	$(PYTHON) $< --school-master "$(INTERIM_DIR)/school_master.csv" \
		--subject-summary "$(PROCESSED_DIR)/school_subject_summary.csv" \
		--blacklist "config/blacklists.yml" \
		--analysis-schools-out "$@" \
		--report-out "$(META_DIR)/feature_coverage_report.csv"

$(PROCESSED_DIR)/nearby_school_accessibility.csv: \
	scripts/06_build_school_accessibility.py \
	src/coursemap/io.py \
	src/coursemap/geo.py \
	$(INTERIM_DIR)/analysis_schools.csv \
	$(PROCESSED_DIR)/school_subject_matrix_binary.csv
	$(PYTHON) $< --school-master "$(INTERIM_DIR)/analysis_schools.csv" \
		--subject-matrix "$(PROCESSED_DIR)/school_subject_matrix_binary.csv" \
		--out-dir "$(PROCESSED_DIR)"

$(PROCESSED_DIR)/joint_curriculum_existing_network.csv: \
	scripts/07_build_joint_network.py \
	src/coursemap/io.py \
	$(INTERIM_DIR)/joint_curriculum.csv
	$(PYTHON) $< --joint "$(INTERIM_DIR)/joint_curriculum.csv" --out "$@"

$(PROCESSED_DIR)/school_features.csv: \
	scripts/08_build_school_features.py \
	src/coursemap/io.py \
	$(INTERIM_DIR)/analysis_schools.csv \
	$(PROCESSED_DIR)/school_subject_summary.csv \
	$(PROCESSED_DIR)/school_domain_supply.csv \
	$(PROCESSED_DIR)/nearby_school_accessibility.csv \
	$(PROCESSED_DIR)/joint_curriculum_existing_network.csv
	$(PYTHON) $< --analysis-schools "$(INTERIM_DIR)/analysis_schools.csv" \
		--subject-summary "$(PROCESSED_DIR)/school_subject_summary.csv" \
		--domain-supply "$(PROCESSED_DIR)/school_domain_supply.csv" \
		--nearby "$(PROCESSED_DIR)/nearby_school_accessibility.csv" \
		--joint-network "$(PROCESSED_DIR)/joint_curriculum_existing_network.csv" \
		--out "$@"

$(BUILD_DIR)/tables/school_sai_result.csv: \
	scripts/09_score_sai.py \
	src/coursemap/io.py \
	src/coursemap/sai.py \
	$(INTERIM_DIR)/analysis_schools.csv \
	$(PROCESSED_DIR)/neis_subjects_standardized.csv
	$(PYTHON) $< --schools "$(INTERIM_DIR)/analysis_schools.csv" \
		--neis-subjects "$(PROCESSED_DIR)/neis_subjects_standardized.csv" \
		--out "$@"

clean:
	rm -rf "$(BUILD_DIR)"

rebuild: clean all
