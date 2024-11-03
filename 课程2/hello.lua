print ("Hello World!")
--单行注释
--[[ 多行注释1
多行注释2
多行注释3
]]--
print('Hello World')





function sysCall_init()
-- This is executed exactly once, the first time this script is executed
    bubbleRobBase=sim.getObjectAssociatedWithScript(sim.handle_self) -- this is bubbleRob's handle
    leftMotor=sim.getObjectHandle("BubbleRob_leftMotor") -- Handle of the left motor
    rightMotor=sim.getObjectHandle("BubbleRob_rightMotor") -- Handle of the right motor
    noseSensor=sim.getObjectHandle("BubbleRob_sensingNose") -- Handle of the proximity sensor
    minMaxSpeed={50*math.pi/180,300*math.pi/180} -- Min and max speeds for each motor
    backUntilTime=-1 -- Tells whether bubbleRob is in forward or backward mode
    robotCollection=sim.createCollection(0)
    sim.addItemToCollection(robotCollection,sim.handle_tree,bubbleRobBase,0)
    distanceSegment=sim.addDrawingObject(sim.drawing_lines,4,0,-1,1,{0,1,0})
    robotTrace=sim.addDrawingObject(sim.drawing_linestrip+sim.drawing_cyclic,2,0,-1,200,{1,1,0},nil,nil,{1,1,0})
    graph=sim.getObjectHandle('BubbleRob_Graph')
    distStream=sim.addGraphStream(graph,'bubbleRob clearance','m',0,{1,0,0})
    -- Create the custom UI:
        xml = '<ui title="'..sim.getObjectName(bubbleRobBase)..' speed" closeable="false" resizeable="false" activate="false">'..[[
        <hslider minimum="0" maximum="100" on-change="speedChange_callback" id="1"/>
        <label text="" style="* {margin-left: 300px;}"/>
        </ui>
        ]]
    ui=simUI.create(xml)
    speed=(minMaxSpeed[1]+minMaxSpeed[2])*0.5
    simUI.setSliderValue(ui,1,100*(speed-minMaxSpeed[1])/(minMaxSpeed[2]-minMaxSpeed[1]))
    -- do some initialization here
end


<ui title="BubbleRob speed" closeable="false" resizeable="false" activate="false">
	<hslider minimum="0" maximum="100" on-change="speedChange_callback" id="1"/>
	<label text="" style="* {margin-left: 300px;}"/>
</ui>

int result, table[7] distanceData, table[2] objectHandlePair = sim.checkDistance( int entity1Handle, int entity2Handle, float threshold=0)
int result=sim.addDrawingObjectItem(int drawingObjectHandle,table[] itemData)