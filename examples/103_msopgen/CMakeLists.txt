execute_process(
    COMMAND ln -s ${ASCEND_HOME_PATH}/tools/op_project_templates/ascendc/customize/cmake .
    COMMAND ln -s ${ASCEND_HOME_PATH}/tools/op_project_templates/ascendc/customize/scripts .
    COMMAND ln -s ${ASCEND_HOME_PATH}/tools/op_project_templates/ascendc/customize/build.sh .
    

    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

execute_process(COMMAND grep -i ^id= /etc/os-release OUTPUT_VARIABLE TEMP)
string(REGEX REPLACE "\n|id=|ID=|\"" "" SYSTEM_NAME ${TEMP})
set(PACKCAGE_NAME custom_opp_${SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR} PARENT_SCOPE)

# add_custom_command(103_msopgen)